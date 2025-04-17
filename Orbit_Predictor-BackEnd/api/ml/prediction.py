"""
ML model prediction module for satellite collision prediction.

This module handles making predictions using trained ML models
for collision probability and risk assessment.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# Feature engineering
from .feature_engineering import extract_features_from_cdm, engineer_advanced_features

# Models
from ..models.ml_model import MLModel, ModelPrediction
from ..models.cdm import CDM


def predict_collision_probability(cdm_id, model_id=None):
    """
    Predict collision probability using a trained ML model.
    
    Args:
        cdm_id: ID of the CDM to predict for
        model_id: Optional ID of a specific ML model to use.
                 If None, use the latest active model.
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Get the CDM
        cdm = CDM.objects.get(id=cdm_id)
        
        # Get the ML model
        if model_id:
            ml_model = MLModel.objects.get(id=model_id, model_type='collision_probability', status='active')
        else:
            # Get the latest active collision probability model
            ml_model = MLModel.objects.filter(
                model_type='collision_probability',
                status='active'
            ).order_by('-created_at').first()
            
        if not ml_model:
            raise ValueError("No active collision probability model found")
        
        # Load the model
        model_data = ml_model.load_model()
        
        # Extract features
        features = extract_features_from_cdm(cdm)
        
        # Create dataframe with single row of features
        df = pd.DataFrame([features])
        
        # Add advanced features
        df = engineer_advanced_features(df)
        
        # Select only the columns used by the model
        if 'feature_cols' in model_data:
            feature_cols = model_data['feature_cols']
            # Handle missing columns with default values
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            X = df[feature_cols]
        else:
            # Fallback to using all features except non-feature columns
            X = df.drop(columns=['cdm_id'], errors='ignore')
        
        # Normalize features
        if 'min_vals' in model_data and 'max_vals' in model_data:
            min_vals = model_data['min_vals']
            max_vals = model_data['max_vals']
            
            X_norm = X.copy()
            for col in X.columns:
                if col in min_vals and col in max_vals:
                    # Skip normalization for binary/categorical features
                    if col in ['sat1_maneuverable', 'sat2_maneuverable']:
                        continue
                        
                    # Min-max normalization
                    if max_vals[col] > min_vals[col]:
                        X_norm[col] = (X[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
                    else:
                        X_norm[col] = 0.5
            
            X = X_norm
        
        # Make prediction
        model = model_data['model']
        probability = float(model.predict(X)[0])
        
        # Get feature importances for this prediction (if SHAP available)
        explanation_data = {}
        if 'metadata' in model_data and 'feature_importances' in model_data['metadata']:
            explanation_data['feature_importances'] = model_data['metadata']['feature_importances']
        
        # Store prediction in database
        prediction = ModelPrediction.objects.create(
            ml_model=ml_model,
            cdm=cdm,
            predicted_probability=probability,
            explanation_data=explanation_data
        )
        
        return {
            'cdm_id': str(cdm.id),
            'model_id': str(ml_model.id),
            'model_name': ml_model.name,
            'model_version': ml_model.version,
            'predicted_probability': probability,
            'prediction_id': str(prediction.id),
            'prediction_time': prediction.prediction_time.isoformat(),
            'explanation_data': explanation_data
        }
        
    except CDM.DoesNotExist:
        raise ValueError(f"CDM with ID {cdm_id} not found")
    except MLModel.DoesNotExist:
        raise ValueError(f"ML model with ID {model_id} not found")
    except Exception as e:
        raise RuntimeError(f"Error predicting collision probability: {str(e)}")


def assess_collision_risk(cdm_id, model_id=None):
    """
    Assess collision risk category using a trained classifier model.
    
    Args:
        cdm_id: ID of the CDM to predict for
        model_id: Optional ID of a specific ML model to use.
                 If None, use the latest active model.
    
    Returns:
        Dictionary with risk assessment results
    """
    try:
        # Get the CDM
        cdm = CDM.objects.get(id=cdm_id)
        
        # Get the ML model
        if model_id:
            ml_model = MLModel.objects.get(id=model_id, model_type='conjunction_risk', status='active')
        else:
            # Get the latest active risk classification model
            ml_model = MLModel.objects.filter(
                model_type='conjunction_risk',
                status='active'
            ).order_by('-created_at').first()
            
        if not ml_model:
            raise ValueError("No active risk classification model found")
        
        # Load the model
        model_data = ml_model.load_model()
        
        # Extract features
        features = extract_features_from_cdm(cdm)
        
        # Create dataframe with single row of features
        df = pd.DataFrame([features])
        
        # Add advanced features
        df = engineer_advanced_features(df)
        
        # Select only the columns used by the model
        if 'feature_cols' in model_data:
            feature_cols = model_data['feature_cols']
            # Handle missing columns with default values
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            X = df[feature_cols]
        else:
            # Fallback to using all features except non-feature columns
            X = df.drop(columns=['cdm_id'], errors='ignore')
        
        # Normalize features
        if 'min_vals' in model_data and 'max_vals' in model_data:
            min_vals = model_data['min_vals']
            max_vals = model_data['max_vals']
            
            X_norm = X.copy()
            for col in X.columns:
                if col in min_vals and col in max_vals:
                    # Skip normalization for binary/categorical features
                    if col in ['sat1_maneuverable', 'sat2_maneuverable']:
                        continue
                        
                    # Min-max normalization
                    if max_vals[col] > min_vals[col]:
                        X_norm[col] = (X[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
                    else:
                        X_norm[col] = 0.5
            
            X = X_norm
        
        # Make prediction
        model = model_data['model']
        
        # Get binary risk class
        risk_class = int(model.predict(X)[0])
        
        # Get probability scores if available
        if hasattr(model, 'predict_proba'):
            proba_array = model.predict_proba(X)[0]
            risk_probabilities = proba_array.tolist()
            
            # Handle case where we might only have probabilities for one class
            if len(risk_probabilities) > 1:
                # We have both classes, use probability of high risk (class 1)
                risk_score = float(risk_probabilities[1])
            else:
                # We only have one class, use the single probability
                risk_score = float(risk_probabilities[0])
                # Determine if this is probability of class 0 or class 1
                if risk_class == 0:  # If predicted class is 0
                    risk_score = 1.0 - risk_score  # Probability of class 1 is inverse
        else:
            risk_score = float(risk_class)
            risk_probabilities = None
        
        # Map risk class to category
        risk_categories = {0: 'Low Risk', 1: 'High Risk'}
        risk_category = risk_categories[risk_class]
        
        # Get feature importances for this prediction
        explanation_data = {}
        if 'metadata' in model_data and 'feature_importances' in model_data['metadata']:
            explanation_data['feature_importances'] = model_data['metadata']['feature_importances']
        
        if risk_probabilities:
            explanation_data['risk_probabilities'] = risk_probabilities
        
        # Store prediction in database
        prediction = ModelPrediction.objects.create(
            ml_model=ml_model,
            cdm=cdm,
            risk_score=risk_score,
            risk_category=risk_category,
            explanation_data=explanation_data
        )
        
        return {
            'cdm_id': str(cdm.id),
            'model_id': str(ml_model.id),
            'model_name': ml_model.name,
            'model_version': ml_model.version,
            'risk_category': risk_category,
            'risk_score': risk_score,
            'prediction_id': str(prediction.id),
            'prediction_time': prediction.prediction_time.isoformat(),
            'explanation_data': explanation_data
        }
        
    except CDM.DoesNotExist:
        raise ValueError(f"CDM with ID {cdm_id} not found")
    except MLModel.DoesNotExist:
        raise ValueError(f"ML model with ID {model_id} not found")
    except Exception as e:
        raise RuntimeError(f"Error assessing collision risk: {str(e)}")


def predict_miss_distance(cdm_id, model_id=None):
    """
    Predict miss distance using a trained ML model.
    
    Args:
        cdm_id: ID of the CDM to predict for
        model_id: Optional ID of a specific ML model to use.
                 If None, use the latest active model.
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Get the CDM
        cdm = CDM.objects.get(id=cdm_id)
        
        # Get the ML model
        if model_id:
            ml_model = MLModel.objects.get(id=model_id, model_type='miss_distance', status='active')
        else:
            # Get the latest active miss distance model
            ml_model = MLModel.objects.filter(
                model_type='miss_distance',
                status='active'
            ).order_by('-created_at').first()
            
        if not ml_model:
            raise ValueError("No active miss distance model found")
        
        # Load the model
        model_data = ml_model.load_model()
        
        # Extract features
        features = extract_features_from_cdm(cdm)
        
        # Create dataframe with single row of features
        df = pd.DataFrame([features])
        
        # Add advanced features
        df = engineer_advanced_features(df)
        
        # Select only the columns used by the model
        if 'feature_cols' in model_data:
            feature_cols = model_data['feature_cols']
            # Handle missing columns with default values
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            X = df[feature_cols]
        else:
            # Fallback to using all features except non-feature columns
            X = df.drop(columns=['cdm_id', 'miss_distance'], errors='ignore')
        
        # Normalize features
        if 'min_vals' in model_data and 'max_vals' in model_data:
            min_vals = model_data['min_vals']
            max_vals = model_data['max_vals']
            
            X_norm = X.copy()
            for col in X.columns:
                if col in min_vals and col in max_vals:
                    # Skip normalization for binary/categorical features
                    if col in ['sat1_maneuverable', 'sat2_maneuverable']:
                        continue
                        
                    # Min-max normalization
                    if max_vals[col] > min_vals[col]:
                        X_norm[col] = (X[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
                    else:
                        X_norm[col] = 0.5
            
            X = X_norm
        
        # Make prediction
        model = model_data['model']
        predicted_miss_distance = float(model.predict(X)[0])
        
        # Ensure miss distance is non-negative
        predicted_miss_distance = max(0.0, predicted_miss_distance)
        
        # Get feature importances for this prediction
        explanation_data = {}
        if 'metadata' in model_data and 'feature_importances' in model_data['metadata']:
            explanation_data['feature_importances'] = model_data['metadata']['feature_importances']
        
        # Store prediction in database
        prediction = ModelPrediction.objects.create(
            ml_model=ml_model,
            cdm=cdm,
            predicted_miss_distance=predicted_miss_distance,
            explanation_data=explanation_data
        )
        
        return {
            'cdm_id': str(cdm.id),
            'model_id': str(ml_model.id),
            'model_name': ml_model.name,
            'model_version': ml_model.version,
            'predicted_miss_distance': predicted_miss_distance,
            'actual_miss_distance': cdm.miss_distance,
            'prediction_id': str(prediction.id),
            'prediction_time': prediction.prediction_time.isoformat(),
            'explanation_data': explanation_data
        }
        
    except CDM.DoesNotExist:
        raise ValueError(f"CDM with ID {cdm_id} not found")
    except MLModel.DoesNotExist:
        raise ValueError(f"ML model with ID {model_id} not found")
    except Exception as e:
        raise RuntimeError(f"Error predicting miss distance: {str(e)}")