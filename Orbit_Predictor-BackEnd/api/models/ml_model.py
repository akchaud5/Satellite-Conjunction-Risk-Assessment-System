"""
Machine Learning Models for Collision Prediction

This module contains the data models for storing and managing machine learning models
for satellite collision probability prediction.
"""

import os
import uuid
import pickle
from django.db import models
from django.conf import settings
from .cdm import CDM

# Define the location to store ML models
ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'api', 'ml_models')
os.makedirs(ML_MODELS_DIR, exist_ok=True)

class MLModel(models.Model):
    """Model to store metadata about trained machine learning models"""
    
    MODEL_TYPES = (
        ('collision_probability', 'Collision Probability Prediction'),
        ('miss_distance', 'Miss Distance Prediction'),
        ('conjunction_risk', 'Conjunction Risk Classification'),
    )
    
    ALGORITHM_CHOICES = (
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('neural_network', 'Neural Network'),
        ('svm', 'Support Vector Machine'),
        ('ensemble', 'Ensemble Model'),
    )
    
    STATUS_CHOICES = (
        ('training', 'Training in Progress'),
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('failed', 'Training Failed'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    model_type = models.CharField(max_length=30, choices=MODEL_TYPES)
    algorithm = models.CharField(max_length=30, choices=ALGORITHM_CHOICES)
    version = models.CharField(max_length=20)
    file_path = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    
    # Model performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)  # Mean Absolute Error
    rmse = models.FloatField(null=True, blank=True)  # Root Mean Squared Error
    
    # Training parameters (stored as JSON in the database)
    training_parameters = models.JSONField(null=True, blank=True)
    
    # Features used by the model
    feature_columns = models.JSONField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.model_type}) - v{self.version}"
    
    def save_model_file(self, model_object):
        """Save the sklearn/pytorch model to disk"""
        if not os.path.exists(ML_MODELS_DIR):
            os.makedirs(ML_MODELS_DIR)
        
        # Create a unique filename based on ID and version
        filename = f"{self.id}_{self.version.replace('.', '_')}.pkl"
        filepath = os.path.join(ML_MODELS_DIR, filename)
        
        # Save the model to disk using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_object, f)
        
        # Update the file_path field and save
        self.file_path = filepath
        self.save()
        
        return filepath
    
    def load_model(self):
        """Load the model from disk"""
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Model file not found at {self.file_path}")
        
        with open(self.file_path, 'rb') as f:
            model = pickle.load(f)
        
        return model


class ModelPrediction(models.Model):
    """Stores prediction results from ML models"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='predictions')
    cdm = models.ForeignKey(CDM, on_delete=models.CASCADE, related_name='ml_predictions')
    predicted_probability = models.FloatField(null=True, blank=True)
    predicted_miss_distance = models.FloatField(null=True, blank=True)
    risk_score = models.FloatField(null=True, blank=True)
    risk_category = models.CharField(max_length=20, null=True, blank=True)
    prediction_time = models.DateTimeField(auto_now_add=True)
    
    # Explanation data (feature importances, SHAP values, etc.)
    explanation_data = models.JSONField(null=True, blank=True)
    
    class Meta:
        ordering = ['-prediction_time']
    
    def __str__(self):
        return f"Prediction for CDM {self.cdm.id} using {self.ml_model.name}"


class TrainingJob(models.Model):
    """Tracks machine learning model training jobs"""
    
    STATUS_CHOICES = (
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ml_model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name='training_jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    log_output = models.TextField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    training_data_count = models.IntegerField(default=0)
    validation_data_count = models.IntegerField(default=0)
    created_by = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, related_name='training_jobs')
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"Training job for {self.ml_model.name} ({self.status})"