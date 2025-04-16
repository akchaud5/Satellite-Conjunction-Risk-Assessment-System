from django.core.management.base import BaseCommand
from django.utils import timezone
from api.models import CDM
import pytz


class Command(BaseCommand):
    help = 'Fixes timezone issues with CDM dates by shifting all dates by a specified number of hours'

    def add_arguments(self, parser):
        parser.add_argument(
            '--hours', type=int, default=-24,
            help='Number of hours to shift dates (default: -24 to fix future dates)'
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be updated without making changes'
        )

    def handle(self, *args, **options):
        hours_shift = options['hours']
        
        # Get all CDMs
        cdms = CDM.objects.all()
        
        if not cdms:
            self.stdout.write(self.style.WARNING("No CDMs found in the database."))
            return
        
        if options['dry_run']:
            self.stdout.write(self.style.WARNING(f"DRY RUN - No changes will be made"))
        
        self.stdout.write(self.style.SUCCESS(f"Shifting all dates by {hours_shift} hours"))
        
        # Update each CDM
        updated_count = 0
        for cdm in cdms:
            # Store original dates for comparison
            original_creation_date = cdm.creation_date
            original_tca = cdm.tca
            
            # Calculate new dates
            new_creation_date = original_creation_date + timezone.timedelta(hours=hours_shift)
            new_tca = original_tca + timezone.timedelta(hours=hours_shift)
            
            if not options['dry_run']:
                # Actually update the dates
                cdm.creation_date = new_creation_date
                cdm.tca = new_tca
                cdm.save()
            
            updated_count += 1
            
            # Show first 3 updates as examples
            if updated_count <= 3:
                self.stdout.write(f"CDM ID: {cdm.id}")
                self.stdout.write(f"  Creation Date: {original_creation_date} → {new_creation_date}")
                self.stdout.write(f"  TCA: {original_tca} → {new_tca}")
                self.stdout.write(f"  Satellites: {cdm.sat1_object_designator} - {cdm.sat2_object_designator}")
                self.stdout.write("")
        
        action = "Would update" if options['dry_run'] else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} {updated_count} CDMs successfully."))