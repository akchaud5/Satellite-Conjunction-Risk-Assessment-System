"""
ML model training module for satellite collision prediction.

This module handles the training of machine learning models
for collision probability prediction and risk assessment.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import xgboost as xgb

# Feature engineering
from .feature_engineering import (
    prepare_dataset_from_cdms,
    engineer_advanced_features,
    normalize_features
)

# Load custom models
from ..models.ml_model import MLModel, TrainingJob, ML_MODELS_DIR
from ..models.cdm import CDM


def train_probability_model(
    training_job_id,
    algorithm='random_forest',
    test_size=0.2,
    random_state=42,
    hyperparams=None
):
    """
    Train a machine learning model to predict collision probability.
    
    Args:
        training_job_id: ID of the training job record
        algorithm: Algorithm to use (random_forest, gradient_boosting, xgboost)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        hyperparams: Optional hyperparameters for the model
        
    Returns:
        Dictionary with training results and metrics
    """
    # Get the training job
    try:
        training_job = TrainingJob.objects.get(id=training_job_id)
        ml_model = training_job.ml_model
    except TrainingJob.DoesNotExist:
        raise ValueError(f"Training job with ID {training_job_id} not found")
    
    # Update job status
    training_job.status = 'running'
    training_job.started_at = datetime.now()
    training_job.save()
    
    try:
        # Get all CDMs with collision data
        cdms_with_collisions = CDM.objects.filter(collisions__isnull=False).distinct()
        
        if not cdms_with_collisions.exists():
            raise ValueError("No CDMs with collision data found for training")
        
        # Prepare dataset
        df = prepare_dataset_from_cdms(cdms_with_collisions, target_variable='probability_of_collision')
        
        # Drop rows with missing target
        df = df.dropna(subset=['probability_of_collision'])
        
        # Feature engineering
        df = engineer_advanced_features(df)
        
        # Select relevant features
        feature_cols = [col for col in df.columns if col not in ['cdm_id', 'probability_of_collision']]
        X = df[feature_cols]
        y = df['probability_of_collision']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normalize features
        exclude_cols = ['cdm_id', 'sat1_maneuverable', 'sat2_maneuverable'] 
        X_train_norm, X_test_norm, min_vals, max_vals = normalize_features(
            X_train, X_test, exclude_cols=exclude_cols
        )
        
        # Initialize model based on algorithm
        if algorithm == 'random_forest':
            if hyperparams:
                model = RandomForestRegressor(**hyperparams)
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=random_state
                )
        elif algorithm == 'gradient_boosting':
            if hyperparams:
                model = GradientBoostingRegressor(**hyperparams)
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
        elif algorithm == 'xgboost':
            if hyperparams:
                model = xgb.XGBRegressor(**hyperparams)
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train model
        training_job.log_output = f"Training {algorithm} model with {len(X_train)} samples...\n"
        training_job.save()
        
        model.fit(X_train_norm, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test_norm)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importances = {
                feature: importance 
                for feature, importance in zip(feature_cols, model.feature_importances_)
            }
            # Sort by importance
            feature_importances = dict(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            feature_importances = {}
        
        # Save model file
        model_data = {
            'model': model,
            'feature_cols': feature_cols,
            'min_vals': min_vals,
            'max_vals': max_vals,
            'metadata': {
                'algorithm': algorithm,
                'training_date': datetime.now().isoformat(),
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                },
                'hyperparameters': model.get_params(),
                'feature_importances': feature_importances
            }
        }
        
        # Update ML model record
        ml_model.accuracy = r2  # Using R² as an accuracy proxy for regression
        ml_model.mae = mae
        ml_model.rmse = rmse
        ml_model.feature_columns = feature_cols
        ml_model.training_parameters = model.get_params()
        ml_model.status = 'active'
        ml_model.save_model_file(model_data)
        ml_model.save()
        
        # Update training job
        training_job.status = 'completed'
        training_job.completed_at = datetime.now()
        training_job.training_data_count = len(X_train)
        training_job.validation_data_count = len(X_test)
        training_job.log_output += f"""
Training completed successfully!
Model metrics:
- MAE: {mae:.6f}
- RMSE: {rmse:.6f}
- R²: {r2:.6f}

Top important features:
{json.dumps({k: float(v) for k, v in list(feature_importances.items())[:5]}, indent=2)}
"""
        training_job.save()
        
        return {
            'success': True,
            'model_id': str(ml_model.id),
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            },
            'feature_importances': feature_importances
        }
        
    except Exception as e:
        # Handle failures
        if training_job:
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.completed_at = datetime.now()
            training_job.save()
            
        if ml_model:
            ml_model.status = 'failed'
            ml_model.save()
            
        raise e


def train_risk_classifier(
    training_job_id,
    algorithm='random_forest',
    threshold=1e-4,  # Threshold for binary classification (high/low risk)
    test_size=0.2,
    random_state=42,
    hyperparams=None
):
    """
    Train a classifier to categorize collision risks.
    
    Args:
        training_job_id: ID of the training job record
        algorithm: Algorithm to use
        threshold: Probability threshold for high risk classification
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        hyperparams: Optional hyperparameters for the model
        
    Returns:
        Dictionary with training results and metrics
    """
    # Get the training job
    try:
        training_job = TrainingJob.objects.get(id=training_job_id)
        ml_model = training_job.ml_model
    except TrainingJob.DoesNotExist:
        raise ValueError(f"Training job with ID {training_job_id} not found")
    
    # Update job status
    training_job.status = 'running'
    training_job.started_at = datetime.now()
    training_job.save()
    
    try:
        # Get all CDMs with collision data
        cdms_with_collisions = CDM.objects.filter(collisions__isnull=False).distinct()
        
        if not cdms_with_collisions.exists():
            raise ValueError("No CDMs with collision data found for training")
        
        # Prepare dataset
        df = prepare_dataset_from_cdms(cdms_with_collisions, target_variable='probability_of_collision')
        
        # Drop rows with missing target
        df = df.dropna(subset=['probability_of_collision'])
        
        # Feature engineering
        df = engineer_advanced_features(df)
        
        # Create binary risk label
        df['high_risk'] = (df['probability_of_collision'] >= threshold).astype(int)
        
        # Select relevant features
        feature_cols = [col for col in df.columns if col not in ['cdm_id', 'probability_of_collision', 'high_risk']]
        X = df[feature_cols]
        y = df['high_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize features
        exclude_cols = ['cdm_id', 'sat1_maneuverable', 'sat2_maneuverable']
        X_train_norm, X_test_norm, min_vals, max_vals = normalize_features(
            X_train, X_test, exclude_cols=exclude_cols
        )
        
        # Initialize model based on algorithm
        if algorithm == 'random_forest':
            if hyperparams:
                model = RandomForestClassifier(**hyperparams)
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=random_state
                )
        elif algorithm == 'gradient_boosting':
            if hyperparams:
                model = GradientBoostingClassifier(**hyperparams)
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
        elif algorithm == 'xgboost':
            if hyperparams:
                model = xgb.XGBClassifier(**hyperparams)
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train model
        training_job.log_output = f"Training {algorithm} classifier with {len(X_train)} samples...\n"
        training_job.save()
        
        model.fit(X_train_norm, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test_norm)
        
        # Safely extract prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_norm)
            # Check if we have probabilities for both classes (0 and 1)
            if y_pred_proba.shape[1] > 1:
                y_pred_prob = y_pred_proba[:, 1]  # Second column is prob of class 1
            else:
                # Handle case where we only have one class in training data
                y_pred_prob = y_pred_proba[:, 0]  # Use the only column available
        else:
            y_pred_prob = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Only calculate AUC if we have both classes and valid probabilities
        if y_pred_prob is not None and len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, y_pred_prob)
        else:
            auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importances = {
                feature: importance 
                for feature, importance in zip(feature_cols, model.feature_importances_)
            }
            # Sort by importance
            feature_importances = dict(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            feature_importances = {}
        
        # Save model file
        model_data = {
            'model': model,
            'feature_cols': feature_cols,
            'min_vals': min_vals,
            'max_vals': max_vals,
            'threshold': threshold,
            'metadata': {
                'algorithm': algorithm,
                'training_date': datetime.now().isoformat(),
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                },
                'hyperparameters': model.get_params(),
                'feature_importances': feature_importances,
                'confusion_matrix': cm.tolist()
            }
        }
        
        # Update ML model record
        ml_model.accuracy = accuracy
        ml_model.precision = precision
        ml_model.recall = recall
        ml_model.f1_score = f1
        ml_model.feature_columns = feature_cols
        ml_model.training_parameters = {
            **model.get_params(),
            'threshold': threshold
        }
        ml_model.status = 'active'
        ml_model.save_model_file(model_data)
        ml_model.save()
        
        # Update training job
        training_job.status = 'completed'
        training_job.completed_at = datetime.now()
        training_job.training_data_count = len(X_train)
        training_job.validation_data_count = len(X_test)
        # Prepare log output with safer string formatting
        log_output = "Training completed successfully!\nModel metrics:\n"
        log_output += f"- Accuracy: {accuracy:.4f}\n"
        log_output += f"- Precision: {precision:.4f}\n"
        log_output += f"- Recall: {recall:.4f}\n"
        log_output += f"- F1 Score: {f1:.4f}\n"
        
        # Safely handle AUC which might be None
        if auc is not None:
            log_output += f"- AUC: {auc:.4f}\n"
        else:
            log_output += "- AUC: N/A\n"
        
        log_output += f"\nConfusion Matrix:\n{cm}\n\n"
        
        # Add top features if available
        if feature_importances:
            top_features = {k: float(v) for k, v in list(feature_importances.items())[:5]}
            log_output += f"Top important features:\n{json.dumps(top_features, indent=2)}"
        else:
            log_output += "Feature importances not available for this model."
            
        training_job.log_output += log_output
        training_job.save()
        
        return {
            'success': True,
            'model_id': str(ml_model.id),
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            },
            'feature_importances': feature_importances,
            'confusion_matrix': cm.tolist()
        }
        
    except Exception as e:
        # Handle failures
        if training_job:
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.completed_at = datetime.now()
            training_job.save()
            
        if ml_model:
            ml_model.status = 'failed'
            ml_model.save()
            
        raise e


def perform_hyperparameter_tuning(
    training_job_id,
    algorithm='random_forest',
    target_type='regression',  # 'regression' or 'classification'
    param_grid=None,
    cv=5,
    scoring=None,
    test_size=0.2,
    random_state=42
):
    """
    Perform hyperparameter tuning for a model.
    
    Args:
        training_job_id: ID of the training job record
        algorithm: Algorithm to tune
        target_type: 'regression' or 'classification'
        param_grid: Grid of parameters to search
        cv: Number of cross-validation folds
        scoring: Scoring metric for grid search
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with best parameters and metrics
    """
    # Get the training job
    try:
        training_job = TrainingJob.objects.get(id=training_job_id)
        ml_model = training_job.ml_model
    except TrainingJob.DoesNotExist:
        raise ValueError(f"Training job with ID {training_job_id} not found")
    
    # Update job status
    training_job.status = 'running'
    training_job.started_at = datetime.now()
    training_job.save()
    
    try:
        # Get all CDMs with collision data
        cdms_with_collisions = CDM.objects.filter(collisions__isnull=False).distinct()
        
        if not cdms_with_collisions.exists():
            raise ValueError("No CDMs with collision data found for training")
        
        # Prepare dataset
        df = prepare_dataset_from_cdms(cdms_with_collisions, target_variable='probability_of_collision')
        
        # Drop rows with missing target
        df = df.dropna(subset=['probability_of_collision'])
        
        # Feature engineering
        df = engineer_advanced_features(df)
        
        # Select relevant features
        feature_cols = [col for col in df.columns if col not in ['cdm_id', 'probability_of_collision']]
        X = df[feature_cols]
        
        if target_type == 'regression':
            y = df['probability_of_collision']
        else:  # classification
            threshold = 1e-4  # Default threshold
            df['high_risk'] = (df['probability_of_collision'] >= threshold).astype(int)
            y = df['high_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if target_type == 'classification' else None
        )
        
        # Normalize features
        exclude_cols = ['cdm_id', 'sat1_maneuverable', 'sat2_maneuverable']
        X_train_norm, X_test_norm, min_vals, max_vals = normalize_features(
            X_train, X_test, exclude_cols=exclude_cols
        )
        
        # Default param grids if none provided
        if param_grid is None:
            if algorithm == 'random_forest' and target_type == 'regression':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif algorithm == 'random_forest' and target_type == 'classification':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced']
                }
            elif algorithm == 'gradient_boosting' and target_type == 'regression':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            elif algorithm == 'gradient_boosting' and target_type == 'classification':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            elif algorithm == 'xgboost' and target_type == 'regression':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            elif algorithm == 'xgboost' and target_type == 'classification':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                    'scale_pos_weight': [1, 3, 5]  # For imbalanced datasets
                }
            else:
                raise ValueError(f"No default param grid for {algorithm} with {target_type}")
                
        # Default scoring metrics
        if scoring is None:
            scoring = 'neg_mean_squared_error' if target_type == 'regression' else 'f1'
        
        # Initialize base model
        if algorithm == 'random_forest':
            if target_type == 'regression':
                base_model = RandomForestRegressor(random_state=random_state)
            else:
                base_model = RandomForestClassifier(random_state=random_state)
        elif algorithm == 'gradient_boosting':
            if target_type == 'regression':
                base_model = GradientBoostingRegressor(random_state=random_state)
            else:
                base_model = GradientBoostingClassifier(random_state=random_state)
        elif algorithm == 'xgboost':
            if target_type == 'regression':
                base_model = xgb.XGBRegressor(random_state=random_state)
            else:
                base_model = xgb.XGBClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Perform grid search
        training_job.log_output = f"Performing hyperparameter tuning for {algorithm} with {len(X_train)} samples...\n"
        training_job.log_output += f"Parameter grid: {json.dumps(param_grid, indent=2)}\n"
        training_job.save()
        
        grid_search = GridSearchCV(
            base_model, param_grid, scoring=scoring, cv=cv, n_jobs=-1,
            verbose=1, return_train_score=True
        )
        
        grid_search.fit(X_train_norm, y_train)
        
        # Get best model and parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Train final model with best parameters
        if algorithm == 'random_forest':
            if target_type == 'regression':
                best_model = RandomForestRegressor(random_state=random_state, **best_params)
            else:
                best_model = RandomForestClassifier(random_state=random_state, **best_params)
        elif algorithm == 'gradient_boosting':
            if target_type == 'regression':
                best_model = GradientBoostingRegressor(random_state=random_state, **best_params)
            else:
                best_model = GradientBoostingClassifier(random_state=random_state, **best_params)
        elif algorithm == 'xgboost':
            if target_type == 'regression':
                best_model = xgb.XGBRegressor(random_state=random_state, **best_params)
            else:
                best_model = xgb.XGBClassifier(random_state=random_state, **best_params)
        
        best_model.fit(X_train_norm, y_train)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test_norm)
        
        if target_type == 'regression':
            # Regression metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
        else:
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = {
                feature: importance 
                for feature, importance in zip(feature_cols, best_model.feature_importances_)
            }
            # Sort by importance
            feature_importances = dict(
                sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            feature_importances = {}
        
        # Save model file
        model_data = {
            'model': best_model,
            'feature_cols': feature_cols,
            'min_vals': min_vals,
            'max_vals': max_vals,
            'metadata': {
                'algorithm': algorithm,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'hyperparameters': best_model.get_params(),
                'feature_importances': feature_importances,
                'cv_results': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in grid_search.cv_results_.items()
                    if k in ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']
                }
            }
        }
        
        # Update ML model record with best parameters and metrics
        ml_model.training_parameters = best_model.get_params()
        ml_model.feature_columns = feature_cols
        ml_model.status = 'active'
        
        if target_type == 'regression':
            ml_model.mae = mae
            ml_model.rmse = rmse
            ml_model.accuracy = r2  # Using R² as an accuracy proxy for regression
        else:
            ml_model.accuracy = accuracy
            ml_model.precision = precision
            ml_model.recall = recall
            ml_model.f1_score = f1
            
        ml_model.save_model_file(model_data)
        ml_model.save()
        
        # Update training job
        training_job.status = 'completed'
        training_job.completed_at = datetime.now()
        training_job.training_data_count = len(X_train)
        training_job.validation_data_count = len(X_test)
        
        if target_type == 'regression':
            training_job.log_output += f"""
Hyperparameter tuning completed successfully!
Best parameters: {json.dumps(best_params, indent=2)}
Best CV score: {best_score:.6f}

Test metrics:
- MAE: {mae:.6f}
- RMSE: {rmse:.6f}
- R²: {r2:.6f}

Top important features:
{json.dumps({k: float(v) for k, v in list(feature_importances.items())[:5]}, indent=2)}
"""
        else:
            training_job.log_output += f"""
Hyperparameter tuning completed successfully!
Best parameters: {json.dumps(best_params, indent=2)}
Best CV score: {best_score:.6f}

Test metrics:
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1 Score: {f1:.4f}

Top important features:
{json.dumps({k: float(v) for k, v in list(feature_importances.items())[:5]}, indent=2)}
"""
        training_job.save()
        
        return {
            'success': True,
            'model_id': str(ml_model.id),
            'best_params': best_params,
            'best_cv_score': best_score,
            'metrics': metrics,
            'feature_importances': feature_importances
        }
        
    except Exception as e:
        # Handle failures
        if training_job:
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.completed_at = datetime.now()
            training_job.save()
            
        if ml_model:
            ml_model.status = 'failed'
            ml_model.save()
            
        raise e