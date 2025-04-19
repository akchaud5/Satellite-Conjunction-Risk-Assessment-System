"""
Management command to create test CDMs.
"""

from django.core.management.base import BaseCommand
from api.models.cdm import CDM
from api.models.collision import Collision
import datetime
import random

class Command(BaseCommand):
    help = 'Create test CDMs for development purposes'

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=5, help='Number of test CDMs to create')

    def handle(self, *args, **options):
        count = options['count']
        
        # Generate random satellite names
        sat_names = [
            "STARLINK", "IRIDIUM", "COSMOS", "ISS", "HUBBLE",
            "NOAA", "GOES", "TERRA", "AQUA", "GPS", "GALILEO"
        ]
        
        created_count = 0
        now = datetime.datetime.now(datetime.timezone.utc)
        
        for i in range(count):
            # Select two different satellites
            sat1_index = random.randint(0, len(sat_names) - 1)
            sat2_index = (sat1_index + random.randint(1, len(sat_names) - 1)) % len(sat_names)
            
            sat1_name = sat_names[sat1_index]
            sat2_name = sat_names[sat2_index]
            
            sat1_designator = f"{sat1_name}-{random.randint(1000, 9999)}"
            sat2_designator = f"{sat2_name}-{random.randint(1000, 9999)}"
            
            # Create a random time of closest approach
            hours_offset = random.randint(-24, 24)
            tca = now + datetime.timedelta(hours=hours_offset)
            
            # Random miss distance between 0.1 and 10 km
            miss_distance = round(random.uniform(0.1, 10.0), 3)
            
            # Create the CDM
            cdm = CDM.objects.create(
                ccsds_cdm_version="1.0",
                creation_date=now,
                originator="TEST-ORIGINATOR",
                message_id=f"TEST-CDM-{i+1}",
                tca=tca,
                miss_distance=miss_distance,
                
                # Satellite 1 info
                sat1_object="PAYLOAD",
                sat1_object_designator=sat1_designator,
                sat1_catalog_name="TEST-CATALOG",
                sat1_object_name=sat1_name,
                sat1_international_designator=f"2020-{random.randint(1, 99):02d}A",
                sat1_object_type="PAYLOAD",
                sat1_operator_organization="TEST-ORG",
                sat1_covariance_method="CALCULATED",
                sat1_maneuverable="YES",
                sat1_reference_frame="EME2000",
                
                # Satellite 1 position (random values in km)
                sat1_x=random.uniform(-7000, 7000),
                sat1_y=random.uniform(-7000, 7000),
                sat1_z=random.uniform(-7000, 7000),
                
                # Satellite 1 velocity (random values in km/s)
                sat1_x_dot=random.uniform(-7, 7),
                sat1_y_dot=random.uniform(-7, 7),
                sat1_z_dot=random.uniform(-7, 7),
                
                # Satellite 1 covariance (small random values)
                sat1_cov_rr=random.uniform(0.001, 0.01),
                sat1_cov_rt=random.uniform(0.001, 0.01),
                sat1_cov_rn=random.uniform(0.001, 0.01),
                sat1_cov_tr=random.uniform(0.001, 0.01),
                sat1_cov_tt=random.uniform(0.001, 0.01),
                sat1_cov_tn=random.uniform(0.001, 0.01),
                sat1_cov_nr=random.uniform(0.001, 0.01),
                sat1_cov_nt=random.uniform(0.001, 0.01),
                sat1_cov_nn=random.uniform(0.001, 0.01),
                
                # Satellite 2 info (similar to satellite 1)
                sat2_object="PAYLOAD",
                sat2_object_designator=sat2_designator,
                sat2_catalog_name="TEST-CATALOG",
                sat2_object_name=sat2_name,
                sat2_international_designator=f"2020-{random.randint(1, 99):02d}B",
                sat2_object_type="PAYLOAD",
                sat2_operator_organization="TEST-ORG",
                sat2_covariance_method="CALCULATED",
                sat2_maneuverable="YES",
                sat2_reference_frame="EME2000",
                
                # Satellite 2 position
                sat2_x=random.uniform(-7000, 7000),
                sat2_y=random.uniform(-7000, 7000),
                sat2_z=random.uniform(-7000, 7000),
                
                # Satellite 2 velocity
                sat2_x_dot=random.uniform(-7, 7),
                sat2_y_dot=random.uniform(-7, 7),
                sat2_z_dot=random.uniform(-7, 7),
                
                # Satellite 2 covariance
                sat2_cov_rr=random.uniform(0.001, 0.01),
                sat2_cov_rt=random.uniform(0.001, 0.01),
                sat2_cov_rn=random.uniform(0.001, 0.01),
                sat2_cov_tr=random.uniform(0.001, 0.01),
                sat2_cov_tt=random.uniform(0.001, 0.01),
                sat2_cov_tn=random.uniform(0.001, 0.01),
                sat2_cov_nr=random.uniform(0.001, 0.01),
                sat2_cov_nt=random.uniform(0.001, 0.01),
                sat2_cov_nn=random.uniform(0.001, 0.01),
                
                # Hard body radius (random between 5 and 20 meters)
                hard_body_radius=round(random.uniform(5, 20), 2),
                
                # Public CDM
                privacy=True
            )
            
            # Create corresponding collision entry with random probability
            probability = random.uniform(0, 0.01)
            Collision.objects.create(
                cdm=cdm,
                probability_of_collision=probability,
                sat1_object_designator=sat1_designator,
                sat2_object_designator=sat2_designator
            )
            
            self.stdout.write(self.style.SUCCESS(
                f'Created CDM {i+1}: {sat1_designator} & {sat2_designator}, '
                f'miss distance: {miss_distance} km, probability: {probability:.8f}'
            ))
            
            created_count += 1
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created {created_count} test CDMs.'))