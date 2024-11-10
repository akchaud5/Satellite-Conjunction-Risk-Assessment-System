from rest_framework import serializers
from .models import Conjunction, Collision, ProbabilityCalc, CDM

class ConjunctionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conjunction
        fields = '__all__'

class CollisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Collision
        fields = '__all__'

class ProbabilityCalcSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProbabilityCalc
        fields = '__all__'

class CDMSerializer(serializers.ModelSerializer):
    class Meta:
        model = CDM
        fields = '__all__'
