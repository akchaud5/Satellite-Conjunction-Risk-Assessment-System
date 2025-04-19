#!/usr/bin/env python3
"""
Create test collision data for ML testing
"""

import os
import sys
import random
from datetime import datetime, timedelta

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'orbit_predictor.settings')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Orbit_Predictor-BackEnd'))

import django
django.setup()

# Import Django models
from api.models import CDM, Collision
from django.utils import timezone

def create_collision_data(num_records=20):
    """Create collision data for existing CDMs"""
    
    cdms = CDM.objects.all()
    if not cdms.exists():
        print("No CDMs found in database. Please seed the database first.")
        return False
    
    cdm_list = list(cdms)
    count = 0
    
    print(f"Creating collision data for {num_records} CDMs")
    
    for i in range(min(num_records, len(cdm_list))):
        cdm = cdm_list[i]
        
        # Skip if collision data already exists
        if Collision.objects.filter(cdm=cdm).exists():
            print(f"CDM {cdm.id} already has collision data, skipping")
            continue
        
        # Calculate a realistic probability based on miss distance
        # Typical formula: Pc ≈ exp(-d²/2σ²) where d is miss distance and σ is position uncertainty
        miss_distance = cdm.miss_distance
        
        # Approximate position uncertainty based on covariance
        uncertainty = (cdm.sat1_cov_rr + cdm.sat2_cov_rr) / 2
        if uncertainty < 0.001:  # Ensure a reasonable minimum
            uncertainty = 0.1
            
        # Calculate probability of collision (capped at reasonable values)
        if miss_distance > 0:
            calc_pc = min(0.9, max(1e-10, random.uniform(0.1, 0.9) * 
                                   (20000.0 / (miss_distance * miss_distance))))
        else:
            calc_pc = random.uniform(0.01, 0.8)  # Random value for testing
        
        # Create collision record
        collision = Collision.objects.create(
            cdm=cdm,
            probability_of_collision=calc_pc,
            sat1_object_designator=cdm.sat1_object_designator,
            sat2_object_designator=cdm.sat2_object_designator
        )
        
        print(f"Created collision data for CDM {cdm.id} with Pc: {calc_pc:.10f}")
        count += 1
    
    print(f"\nCreated {count} collision records")
    return True

if __name__ == "__main__":
    create_collision_data(20)