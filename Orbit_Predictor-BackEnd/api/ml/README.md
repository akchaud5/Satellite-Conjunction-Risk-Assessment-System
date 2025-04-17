# Machine Learning Module for Satellite Collision Prediction

This module provides machine learning capabilities to enhance the satellite collision prediction system with AI-driven insights.

## Features

- **Collision Probability Prediction**: ML models trained to predict the probability of collision between two satellites
- **Conjunction Risk Classification**: Binary classifiers to categorize conjunctions as high or low risk
- **Miss Distance Prediction**: Models to predict the miss distance between two satellites
- **Hyperparameter Tuning**: Automated tuning of model parameters for optimal performance
- **Model Management**: Database models for tracking ML models, training jobs, and predictions
- **Explainable AI**: Feature importance analysis to understand prediction drivers

## Components

### Data Models

- **MLModel**: Stores metadata about trained machine learning models
- **TrainingJob**: Tracks the progress and results of model training sessions
- **ModelPrediction**: Records predictions made by ML models for CDMs

### Core Functionality

- **feature_engineering.py**: Extracts and transforms features from CDM data
- **training.py**: Handles model training and hyperparameter tuning
- **prediction.py**: Makes predictions using trained models

### Available Algorithms

- Random Forest (Classification/Regression)
- Gradient Boosting (Classification/Regression)
- XGBoost (Classification/Regression)

## API Endpoints

- `GET/POST /api/ml/models/`: List and create ML models
- `GET/PUT/DELETE /api/ml/models/{id}/`: Retrieve, update, or delete a model
- `GET/POST /api/ml/training/`: List and create training jobs
- `GET /api/ml/training/{id}/`: Retrieve details about a training job
- `POST /api/ml/predict/`: Make a prediction using a trained model
- `GET /api/ml/predictions/`: List predictions for a CDM
- `POST /api/ml/compare/`: Compare predictions from different models

## Management Commands

```bash
# Train a new collision probability model using Random Forest
python manage.py train_ml_model --model-type collision_probability --algorithm random_forest

# Train a risk classifier with hyperparameter tuning
python manage.py train_ml_model --model-type conjunction_risk --tune

# Train a miss distance prediction model with a specific name and version
python manage.py train_ml_model --model-type miss_distance --name "Enhanced Miss Distance Predictor" --version "2.0.0"
```

## System Requirements

- scikit-learn (1.2.0 or higher)
- pandas (1.5.0 or higher)
- numpy (1.24.0 or higher)
- xgboost (1.7.0 or higher)
- joblib (1.3.0 or higher)

## Integration with Existing Components

The ML module integrates with the existing collision prediction system by:

1. Using CDM data as input for training models
2. Applying both analytical and ML-based approaches to collision probability estimation
3. Providing additional insights through feature importance analysis
4. Enhancing decision-making with risk categorization

## Future Enhancements

- Neural network models for more complex patterns
- Time-series forecasting for conjunction prediction
- Ensemble methods combining multiple model types
- Automated model retraining based on performance metrics