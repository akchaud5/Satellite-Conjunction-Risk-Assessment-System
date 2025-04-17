"""
Serializers for ML models and related objects.
"""

from rest_framework import serializers
from ..models.ml_model import MLModel, TrainingJob, ModelPrediction
from ..models.cdm import CDM
from ..serializers.cdm_serializer import CDMSerializer


class MLModelSerializer(serializers.ModelSerializer):
    """Serializer for MLModel"""
    
    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'description', 'model_type', 'algorithm',
            'version', 'file_path', 'created_at', 'updated_at', 'status',
            'accuracy', 'precision', 'recall', 'f1_score', 'mae', 'rmse',
            'training_parameters', 'feature_columns'
        ]
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'accuracy', 'precision',
            'recall', 'f1_score', 'mae', 'rmse'
        ]


class TrainingJobSerializer(serializers.ModelSerializer):
    """Serializer for TrainingJob"""
    
    ml_model = MLModelSerializer(read_only=True)
    ml_model_id = serializers.UUIDField(write_only=True)
    created_by_name = serializers.SerializerMethodField()
    
    class Meta:
        model = TrainingJob
        fields = [
            'id', 'ml_model', 'ml_model_id', 'status', 'started_at',
            'completed_at', 'log_output', 'error_message',
            'training_data_count', 'validation_data_count',
            'created_by', 'created_by_name'
        ]
        read_only_fields = [
            'id', 'started_at', 'completed_at', 'log_output',
            'error_message', 'training_data_count', 'validation_data_count',
            'created_by'
        ]
    
    def get_created_by_name(self, obj):
        """Get created_by user's name"""
        return obj.created_by.email if obj.created_by else None
    
    def create(self, validated_data):
        """Create a TrainingJob"""
        ml_model_id = validated_data.pop('ml_model_id')
        user = self.context['request'].user
        
        # Create the TrainingJob with the provided ml_model_id and the current user
        training_job = TrainingJob.objects.create(
            ml_model_id=ml_model_id,
            created_by=user,
            **validated_data
        )
        
        return training_job


class ModelPredictionSerializer(serializers.ModelSerializer):
    """Serializer for ModelPrediction"""
    
    ml_model = MLModelSerializer(read_only=True)
    cdm = CDMSerializer(read_only=True)
    
    class Meta:
        model = ModelPrediction
        fields = [
            'id', 'ml_model', 'cdm', 'predicted_probability',
            'predicted_miss_distance', 'risk_score', 'risk_category',
            'prediction_time', 'explanation_data'
        ]
        read_only_fields = fields
        
        
class PredictionRequestSerializer(serializers.Serializer):
    """
    Serializer for prediction request data
    """
    cdm_id = serializers.UUIDField(required=True)
    model_id = serializers.UUIDField(required=False)
    prediction_type = serializers.CharField(
        default='collision_probability',
        required=False
    )
    
    def validate_prediction_type(self, value):
        """Validate prediction_type"""
        valid_types = ['collision_probability', 'conjunction_risk', 'miss_distance']
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Invalid prediction type. Must be one of: {', '.join(valid_types)}"
            )
        return value
    
    def validate_cdm_id(self, value):
        """Validate cdm_id exists"""
        try:
            CDM.objects.get(id=value)
        except CDM.DoesNotExist:
            raise serializers.ValidationError(f"CDM with ID {value} does not exist")
        return value
    
    def validate_model_id(self, value):
        """Validate model_id exists and is active"""
        if value:
            try:
                model = MLModel.objects.get(id=value)
                if model.status != 'active':
                    raise serializers.ValidationError(
                        f"ML model with ID {value} is not active (status: {model.status})"
                    )
            except MLModel.DoesNotExist:
                raise serializers.ValidationError(f"ML model with ID {value} does not exist")
        return value


class TrainingRequestSerializer(serializers.Serializer):
    """
    Serializer for ML model training request data
    """
    ml_model = serializers.UUIDField(required=False)
    model_name = serializers.CharField(required=False)
    model_type = serializers.CharField(required=False, default='collision_probability')
    algorithm = serializers.CharField(required=False, default='random_forest')
    description = serializers.CharField(required=False, allow_blank=True)
    version = serializers.CharField(required=False, default='1.0.0')
    tune = serializers.BooleanField(required=False, default=False)
    test_size = serializers.FloatField(required=False, default=0.2)
    
    def validate_model_type(self, value):
        """Validate model_type"""
        valid_types = ['collision_probability', 'conjunction_risk', 'miss_distance']
        if value not in valid_types:
            raise serializers.ValidationError(
                f"Invalid model type. Must be one of: {', '.join(valid_types)}"
            )
        return value
    
    def validate_algorithm(self, value):
        """Validate algorithm"""
        valid_algorithms = ['random_forest', 'gradient_boosting', 'xgboost']
        if value not in valid_algorithms:
            raise serializers.ValidationError(
                f"Invalid algorithm. Must be one of: {', '.join(valid_algorithms)}"
            )
        return value
    
    def validate_test_size(self, value):
        """Validate test_size is between 0 and 1"""
        if value <= 0 or value >= 1:
            raise serializers.ValidationError("test_size must be between 0 and 1")
        return value
    
    def validate(self, data):
        """Validate complete request data"""
        if 'ml_model' not in data and 'model_name' not in data:
            raise serializers.ValidationError(
                "Either ml_model ID or model_name must be provided"
            )
        return data