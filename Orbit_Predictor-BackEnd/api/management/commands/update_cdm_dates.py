from django.core.management.base import BaseCommand
from django.utils import timezone
from api.models import CDM
from datetime import datetime, timedelta
import pytz


class Command(BaseCommand):
    help = 'Updates CDM dates to be relative to the current time'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days', type=int, default=None,
            help='Specific number of days to shift dates (optional)'
        )
        parser.add_argument(
            '--reference-date', type=str, default=None,
            help='Reference date to calculate shift from (format: YYYY-MM-DD)'
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be updated without making changes'
        )

    def handle(self, *args, **options):
        # Get all CDMs
        cdms = CDM.objects.all()
        
        if not cdms:
            self.stdout.write(self.style.WARNING("No CDMs found in the database."))
            return
        
        # Get the specified or earliest CDM creation date as reference
        if options['reference_date']:
            try:
                reference_date = datetime.strptime(options['reference_date'], '%Y-%m-%d')
                reference_date = pytz.UTC.localize(reference_date)
            except ValueError:
                self.stdout.write(self.style.ERROR("Invalid date format. Use YYYY-MM-DD."))
                return
        else:
            earliest_cdm = min(cdms, key=lambda x: x.creation_date)
            reference_date = earliest_cdm.creation_date
        
        # Calculate the time difference to shift dates
        now = timezone.now()
        
        if options['days'] is not None:
            # Use specified number of days instead of calculating from reference
            time_shift = timedelta(days=options['days'])
        else:
            # Calculate shift based on the difference between now and reference
            time_shift = now - reference_date
        
        self.stdout.write(self.style.SUCCESS(f"Reference date from CDM database: {reference_date}"))
        self.stdout.write(self.style.SUCCESS(f"Current time: {now}"))
        self.stdout.write(self.style.SUCCESS(f"Shifting all dates by: {time_shift}"))
        
        if options['dry_run']:
            self.stdout.write(self.style.WARNING("DRY RUN - No changes will be made"))
            
        # Update each CDM
        updated_count = 0
        for cdm in cdms:
            # Store original dates for comparison
            original_creation_date = cdm.creation_date
            original_tca = cdm.tca
            
            # Calculate new dates
            new_creation_date = original_creation_date + time_shift
            new_tca = original_tca + time_shift
            
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