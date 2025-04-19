"""
Management command to create test ML models.
"""

from django.core.management.base import BaseCommand
from api.models.ml_model import MLModel
import uuid

class Command(BaseCommand):
    help = 'Create test ML models for development purposes'

    def add_arguments(self, parser):
        parser.add_argument('--count', type=int, default=3, help='Number of test models to create for each type')

    def handle(self, *args, **options):
        count = options['count']
        
        # Delete existing test models if they have 'test' in their name
        existing_test_models = MLModel.objects.filter(name__icontains='test')
        if existing_test_models.exists():
            count_deleted = existing_test_models.count()
            existing_test_models.delete()
            self.stdout.write(self.style.WARNING(f'Deleted {count_deleted} existing test ML models.'))
        
        model_types = ['collision_probability', 'conjunction_risk', 'miss_distance']
        algorithms = ['random_forest', 'gradient_boosting', 'xgboost']
        
        created_count = 0
        
        # Create test models
        for model_type in model_types:
            for i in range(count):
                algorithm = algorithms[i % len(algorithms)]
                model_name = f"Test {model_type.replace('_', ' ').title()} Model {uuid.uuid4().hex[:8]}"
                
                model = MLModel.objects.create(
                    name=model_name,
                    description=f"Test model for {model_type.replace('_', ' ')}",
                    model_type=model_type,
                    algorithm=algorithm,
                    version="1.0.0-test",
                    status="active"
                )
                
                # Add some fake performance metrics
                if model_type == 'collision_probability':
                    model.mae = 0.001 * (i + 1)
                    model.rmse = 0.002 * (i + 1)
                elif model_type == 'conjunction_risk':
                    model.accuracy = 0.95 - (0.05 * i)
                    model.precision = 0.94 - (0.05 * i)
                    model.recall = 0.93 - (0.05 * i)
                    model.f1_score = 0.92 - (0.05 * i)
                
                model.save()
                created_count += 1
                
                self.stdout.write(self.style.SUCCESS(
                    f'Created model: {model_name} ({model_type}, {algorithm})'
                ))
        
        self.stdout.write(self.style.SUCCESS(f'Successfully created {created_count} test ML models.'))