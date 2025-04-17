"""
Management command for training ML models.
"""

import uuid
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from api.models.ml_model import MLModel, TrainingJob
from api.ml.training import (
    train_probability_model,
    train_risk_classifier,
    perform_hyperparameter_tuning
)


class Command(BaseCommand):
    help = 'Train machine learning models for collision prediction'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-type',
            type=str,
            choices=['collision_probability', 'conjunction_risk', 'miss_distance'],
            default='collision_probability',
            help='Type of ML model to train'
        )
        parser.add_argument(
            '--algorithm',
            type=str,
            choices=['random_forest', 'gradient_boosting', 'xgboost'],
            default='random_forest',
            help='ML algorithm to use'
        )
        parser.add_argument(
            '--name',
            type=str,
            help='Name for the ML model'
        )
        parser.add_argument(
            '--description',
            type=str,
            help='Description for the ML model'
        )
        parser.add_argument(
            '--version',
            type=str,
            default='1.0.0',
            help='Version of the ML model'
        )
        parser.add_argument(
            '--tune',
            action='store_true',
            help='Perform hyperparameter tuning'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Portion of data to use for testing (0.0-1.0)'
        )

    def handle(self, *args, **options):
        model_type = options['model_type']
        algorithm = options['algorithm']
        name = options['name']
        description = options['description']
        version = options['version']
        tune = options['tune']
        test_size = options['test_size']
        
        # Generate name if not provided
        if not name:
            name = f"{model_type.replace('_', ' ').title()} - {algorithm.replace('_', ' ').title()}"
            
        # Create ML model record
        ml_model = MLModel.objects.create(
            name=name,
            description=description or f"Trained {algorithm} model for {model_type}",
            model_type=model_type,
            algorithm=algorithm,
            version=version,
            status='training'
        )
        
        # Create training job
        training_job = TrainingJob.objects.create(
            ml_model=ml_model,
            status='queued'
        )
        
        self.stdout.write(self.style.SUCCESS(f"Created ML model: {ml_model.name} (ID: {ml_model.id})"))
        self.stdout.write(self.style.SUCCESS(f"Created training job: {training_job.id}"))
        
        try:
            if tune:
                self.stdout.write(self.style.WARNING("Starting hyperparameter tuning (this may take some time)..."))
                
                # Determine target type based on model_type
                target_type = 'regression' if model_type in ['collision_probability', 'miss_distance'] else 'classification'
                
                result = perform_hyperparameter_tuning(
                    training_job_id=training_job.id,
                    algorithm=algorithm,
                    target_type=target_type,
                    test_size=test_size
                )
                
                self.stdout.write(self.style.SUCCESS("Hyperparameter tuning completed successfully!"))
                self.stdout.write(f"Best parameters: {result['best_params']}")
                self.stdout.write(f"Best CV score: {result['best_cv_score']:.6f}")
                
            else:
                self.stdout.write(self.style.WARNING(f"Training {algorithm} model for {model_type}..."))
                
                if model_type == 'collision_probability':
                    result = train_probability_model(
                        training_job_id=training_job.id,
                        algorithm=algorithm,
                        test_size=test_size
                    )
                elif model_type == 'conjunction_risk':
                    result = train_risk_classifier(
                        training_job_id=training_job.id,
                        algorithm=algorithm,
                        test_size=test_size
                    )
                else:
                    # Add support for miss_distance models
                    raise CommandError(f"Training for model type '{model_type}' not yet implemented")
                
                self.stdout.write(self.style.SUCCESS("Training completed successfully!"))
            
            # Display metrics
            self.stdout.write("Model metrics:")
            for metric, value in result['metrics'].items():
                self.stdout.write(f"- {metric}: {value:.6f}")
            
            # Display feature importance
            self.stdout.write("\nTop 5 important features:")
            for i, (feature, importance) in enumerate(list(result['feature_importances'].items())[:5]):
                self.stdout.write(f"{i+1}. {feature}: {importance:.6f}")
            
            self.stdout.write(self.style.SUCCESS(f"\nModel ID: {ml_model.id}"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error training model: {str(e)}"))
            ml_model.status = 'failed'
            ml_model.save()
            
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.completed_at = timezone.now()
            training_job.save()
            
            raise CommandError(str(e))