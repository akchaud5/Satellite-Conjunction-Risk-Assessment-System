"""
Views for ML model management and predictions.
"""

from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.utils import timezone
import uuid

from ..models.ml_model import MLModel, TrainingJob, ModelPrediction
from ..models.cdm import CDM
from ..serializers.ml_serializer import (
    MLModelSerializer, 
    TrainingJobSerializer,
    ModelPredictionSerializer
)
from ..permissions import IsAdmin, IsCollisionAnalyst
from ..ml.prediction import (
    predict_collision_probability,
    assess_collision_risk,
    predict_miss_distance
)
from ..ml.training import (
    train_probability_model,
    train_risk_classifier,
    perform_hyperparameter_tuning
)


class MLModelListCreateView(generics.ListCreateAPIView):
    """
    List and create ML models
    """
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [IsAuthenticated, IsCollisionAnalyst]
    
    def get_queryset(self):
        """Filter query by status and model_type if provided"""
        queryset = MLModel.objects.all()
        status = self.request.query_params.get('status')
        model_type = self.request.query_params.get('model_type')
        
        if status:
            queryset = queryset.filter(status=status)
        if model_type:
            queryset = queryset.filter(model_type=model_type)
            
        return queryset


class MLModelDetailView(generics.RetrieveUpdateDestroyAPIView):
    """
    Retrieve, update or delete an ML model
    """
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [IsAuthenticated, IsCollisionAnalyst]


class TrainingJobListCreateView(generics.ListCreateAPIView):
    """
    List and create training jobs
    """
    queryset = TrainingJob.objects.all()
    serializer_class = TrainingJobSerializer
    permission_classes = [IsAuthenticated, IsCollisionAnalyst]
    
    def get_queryset(self):
        """Filter query by status and ml_model if provided"""
        queryset = TrainingJob.objects.all()
        status = self.request.query_params.get('status')
        ml_model_id = self.request.query_params.get('ml_model')
        
        if status:
            queryset = queryset.filter(status=status)
        if ml_model_id:
            queryset = queryset.filter(ml_model__id=ml_model_id)
            
        return queryset
    
    def create(self, request, *args, **kwargs):
        """Create a new training job"""
        ml_model_id = request.data.get('ml_model')
        algorithm = request.data.get('algorithm')
        
        if not ml_model_id:
            # Create a new ML model if one isn't specified
            model_name = request.data.get('model_name', 'ML Model')
            model_type = request.data.get('model_type', 'collision_probability')
            model_algorithm = algorithm or 'random_forest'
            model_version = request.data.get('version', '1.0.0')
            model_description = request.data.get('description', '')
            
            ml_model = MLModel.objects.create(
                name=model_name,
                description=model_description,
                model_type=model_type,
                algorithm=model_algorithm,
                version=model_version,
                status='training'
            )
            ml_model_id = ml_model.id
        else:
            ml_model = get_object_or_404(MLModel, id=ml_model_id)
            ml_model.status = 'training'
            ml_model.save()
        
        # Create training job
        training_job = TrainingJob.objects.create(
            ml_model_id=ml_model_id,
            status='queued',
            created_by=request.user
        )
        
        # Queue the training task (in a real system, this would be an async celery task)
        # For now, we'll run it synchronously
        try:
            if ml_model.model_type == 'collision_probability':
                result = train_probability_model(
                    training_job_id=training_job.id,
                    algorithm=algorithm or ml_model.algorithm
                )
            elif ml_model.model_type == 'conjunction_risk':
                result = train_risk_classifier(
                    training_job_id=training_job.id,
                    algorithm=algorithm or ml_model.algorithm
                )
            else:
                return Response(
                    {"error": f"Training for model type '{ml_model.model_type}' not implemented"},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            return Response({
                "message": "Training job completed successfully",
                "training_job_id": str(training_job.id),
                "ml_model_id": str(ml_model.id),
                "result": result
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            training_job.status = 'failed'
            training_job.error_message = str(e)
            training_job.save()
            
            ml_model.status = 'failed'
            ml_model.save()
            
            return Response(
                {"error": f"Training failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TrainingJobDetailView(generics.RetrieveAPIView):
    """
    Retrieve a training job
    """
    queryset = TrainingJob.objects.all()
    serializer_class = TrainingJobSerializer
    permission_classes = [IsAuthenticated, IsCollisionAnalyst]


class MLPredictionView(APIView):
    """
    Make predictions using ML models
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        """Make a prediction for a CDM"""
        cdm_id = request.data.get('cdm_id')
        model_id = request.data.get('model_id')
        prediction_type = request.data.get('prediction_type', 'collision_probability')
        
        if not cdm_id:
            return Response(
                {"error": "CDM ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            if prediction_type == 'collision_probability':
                result = predict_collision_probability(cdm_id, model_id)
            elif prediction_type == 'conjunction_risk':
                result = assess_collision_risk(cdm_id, model_id)
            elif prediction_type == 'miss_distance':
                result = predict_miss_distance(cdm_id, model_id)
            else:
                return Response(
                    {"error": f"Prediction type '{prediction_type}' not supported"},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            return Response(result)
            
        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {"error": f"Prediction failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelPredictionListView(generics.ListAPIView):
    """
    List predictions for a CDM
    """
    serializer_class = ModelPredictionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter predictions by CDM ID"""
        cdm_id = self.request.query_params.get('cdm_id')
        
        if cdm_id:
            return ModelPrediction.objects.filter(cdm__id=cdm_id)
        else:
            return ModelPrediction.objects.none()


class ModelComparisonView(APIView):
    """
    Compare predictions from different ML models
    """
    permission_classes = [IsAuthenticated, IsCollisionAnalyst]
    
    def post(self, request, *args, **kwargs):
        """Compare predictions from different models for the same CDM"""
        cdm_id = request.data.get('cdm_id')
        model_ids = request.data.get('model_ids', [])
        prediction_type = request.data.get('prediction_type', 'collision_probability')
        
        if not cdm_id:
            return Response(
                {"error": "CDM ID is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        if not model_ids:
            return Response(
                {"error": "At least one model ID is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = []
        
        try:
            for model_id in model_ids:
                if prediction_type == 'collision_probability':
                    result = predict_collision_probability(cdm_id, model_id)
                elif prediction_type == 'conjunction_risk':
                    result = assess_collision_risk(cdm_id, model_id)
                elif prediction_type == 'miss_distance':
                    result = predict_miss_distance(cdm_id, model_id)
                else:
                    return Response(
                        {"error": f"Prediction type '{prediction_type}' not supported"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                    
                results.append(result)
                
            return Response({
                "cdm_id": cdm_id,
                "prediction_type": prediction_type,
                "results": results
            })
            
        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {"error": f"Comparison failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )