import requests
from django.core.management.base import BaseCommand
from api.models import CDM

class Command(BaseCommand):
    help = 'Checks for inactive satellites and removes CDMs with inactive satellites'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Perform a dry run without deleting any CDMs',
        )

    def handle(self, *args, **options):
        # NASA API key
        NASA_API_KEY = "bFDWy9zOJ2zOnjLbd8rz0eZtVcWwkyT4XUrC3II5"
        dry_run = options['dry_run']

        # Get all CDMs
        all_cdms = CDM.objects.all()
        self.stdout.write(self.style.SUCCESS(f"Total CDMs in database: {all_cdms.count()}"))
        
        # Get unique NORAD IDs
        sat1_norad_ids = set([cdm.sat1_object_designator for cdm in all_cdms])
        sat2_norad_ids = set([cdm.sat2_object_designator for cdm in all_cdms])
        all_norad_ids = sat1_norad_ids.union(sat2_norad_ids)
        
        self.stdout.write(f"Unique satellites (NORAD IDs): {all_norad_ids}")
        
        # Check which satellites are active
        inactive_satellites = []
        for norad_id in all_norad_ids:
            active = self.check_satellite_active(norad_id, NASA_API_KEY)
            if not active:
                inactive_satellites.append(norad_id)
        
        self.stdout.write(self.style.WARNING(f"\nInactive satellites: {inactive_satellites}"))
        
        # Delete CDMs containing inactive satellites
        if inactive_satellites:
            to_delete = []
            
            for cdm in all_cdms:
                if (cdm.sat1_object_designator in inactive_satellites or 
                    cdm.sat2_object_designator in inactive_satellites):
                    to_delete.append(cdm.id)
                    self.stdout.write(
                        f"Marking for deletion: CDM {cdm.id} "
                        f"({cdm.sat1_object} [{cdm.sat1_object_designator}] ↔ "
                        f"{cdm.sat2_object} [{cdm.sat2_object_designator}])"
                    )
            
            if to_delete:
                if dry_run:
                    self.stdout.write(
                        self.style.WARNING(
                            f"\nDRY RUN: Would delete {len(to_delete)} CDMs containing inactive satellites."
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f"\nDeleting {len(to_delete)} CDMs containing inactive satellites."
                        )
                    )
                    CDM.objects.filter(id__in=to_delete).delete()
                    self.stdout.write(self.style.SUCCESS(f"Deleted {len(to_delete)} CDMs."))
        else:
            self.stdout.write(self.style.SUCCESS("All satellites are active! No CDMs need to be deleted."))

    def check_satellite_active(self, norad_id, api_key):
        """Check if satellite is active by trying to fetch TLE data."""
        self.stdout.write(f"Checking NORAD ID: {norad_id}")
        
        # Try NASA source
        url = f"https://tle.ivanstanojevic.me/api/tle/{norad_id}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "SatelliteCollisionPredictor/1.0",
            "X-Api-Key": api_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                self.stdout.write(self.style.SUCCESS(f"  ✅ Active - TLE data found"))
                return True
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Error checking NASA API: {str(e)}"))
        
        # Try CelesTrak as backup
        url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and "No GP data found" not in response.text:
                self.stdout.write(self.style.SUCCESS(f"  ✅ Active - TLE data found from CelesTrak"))
                return True
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"  ❌ Error checking CelesTrak: {str(e)}"))
        
        self.stdout.write(self.style.ERROR(f"  ❌ Inactive - No TLE data found"))
        return False