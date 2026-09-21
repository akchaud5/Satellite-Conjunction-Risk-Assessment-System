# api/serializers.py

import jwt
import datetime
from django.conf import settings
from rest_framework import serializers

from ..models import User, CDM

SPACE_AGENCY_DOMAINS = [
    "asc-csa.gc.ca",
    "nasa.gov",
    "esa.int",
    "roscosmos.ru",
    "cnsa.gov.cn",
    "isro.gov.in",
    "jaxa.jp",
    "gov.uk/government/organisations/uk-space-agency",
    "cnes.fr",
    "dlr.de",
    "asi.it",
    "aeb.gov.br",
    "kari.re.kr",
    "space.gov.ae",
    "australianspaceagency.gov.au",
    "space.gov.il"
]


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    registration_code = serializers.CharField(write_only=True, required=False, allow_blank=True)
    
    # NEW: Allow setting 'interested_cdms' by passing a list of CDM IDs
    interested_cdms = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=CDM.objects.all(),
        required=False
    )

    class Meta:
        model = User
        fields = [
            'id', 
            'email', 
            'password', 
            'role', 
            'registration_code', 
            'created_at',
            'interested_cdms'   # NEW FIELD
        ]
        # 'role' is read-only here. It used to be writable, which meant any
        # authenticated user could PATCH their own record with
        # {"role": "admin"} and escalate to administrator -- UserViewSet
        # deliberately lets users edit themselves. Role is derived from the
        # email domain and registration code on create (see create() below);
        # administrators change it through AdminUserSerializer instead.
        read_only_fields = ['id', 'created_at', 'role']

    def validate_registration_code(self, value):
        """
        Validate the registration code if provided.
        """
        if value and value != settings.ADMIN_REGISTRATION_CODE:
            raise serializers.ValidationError('Invalid registration code.')
        return value

    def create(self, validated_data):
        registration_code = validated_data.pop('registration_code', None)
        email = validated_data.get('email', '')
        domain = email.split('@')[-1].lower()

        # Determine role based on email domain and registration code
        role = 'user'  # Default role
        if domain in SPACE_AGENCY_DOMAINS:
            if registration_code:
                role = 'admin'
            else:
                role = 'collision_analyst'

        validated_data['role'] = role

        # Extract password separately
        password = validated_data.pop('password')
        
        # Extract any CDM IDs passed in
        interested_cdms = validated_data.pop('interested_cdms', [])

        user = User.objects.create_user(password=password, **validated_data)

        # If the user provided interested CDMs during registration, set them
        if interested_cdms:
            user.interested_cdms.set(interested_cdms)

        return user

    def update(self, instance, validated_data):
        # Handle interested_cdms updates
        interested_cdms = validated_data.pop('interested_cdms', None)
        if interested_cdms is not None:
            instance.interested_cdms.set(interested_cdms)

        # Handle password updates (if provided)
        password = validated_data.pop('password', None)
        if password:
            instance.set_password(password)

        return super().update(instance, validated_data)


class AdminUserSerializer(UserSerializer):
    """UserSerializer with 'role' writable, for administrators only.

    UserViewSet selects this serializer when the requesting user is an admin,
    so role assignment stays an admin capability without exposing it to the
    self-service update path.
    """

    class Meta(UserSerializer.Meta):
        read_only_fields = ['id', 'created_at']

    def validate_role(self, value):
        valid = {choice[0] for choice in User.ROLE_CHOICES}
        if value not in valid:
            raise serializers.ValidationError(
                f"Invalid role. Must be one of: {', '.join(sorted(valid))}."
            )
        return value


def _issue_token(user, token_type, lifetime):
    """Mint a signed JWT carrying an explicit token_type claim.

    The token_type claim is what stops a refresh token being replayed as an
    access token: JWTAuthentication only accepts type 'access', and the refresh
    endpoint only accepts type 'refresh'. Without it both tokens were
    interchangeable, silently giving every session a 7-day access lifetime.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        'user_id': str(user.id),
        'role': user.role,
        'token_type': token_type,
        'exp': now + lifetime,
        'iat': now,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    # Changed to 'access' for consistency with frontend
    access = serializers.CharField(read_only=True)
    refresh_token = serializers.CharField(read_only=True)

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            raise serializers.ValidationError('Invalid credentials')

        if not user.check_password(password):
            raise serializers.ValidationError('Invalid credentials')

        return {
            'access': _issue_token(
                user, 'access', settings.JWT_ACCESS_EXPIRATION_DELTA
            ),
            'refresh_token': _issue_token(
                user, 'refresh', settings.JWT_REFRESH_EXPIRATION_DELTA
            ),
        }


class RefreshTokenSerializer(serializers.Serializer):
    refresh_token = serializers.CharField()

    def validate(self, data):
        refresh_token = data.get('refresh_token')

        try:
            payload = jwt.decode(
                refresh_token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except jwt.ExpiredSignatureError:
            raise serializers.ValidationError('Refresh token has expired.')
        except jwt.InvalidTokenError:
            raise serializers.ValidationError('Invalid refresh token.')

        if payload.get('token_type') != 'refresh':
            raise serializers.ValidationError('Invalid refresh token.')

        user_id = payload.get('user_id')

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise serializers.ValidationError('User does not exist.')

        return {
            'access': _issue_token(
                user, 'access', settings.JWT_ACCESS_EXPIRATION_DELTA
            )
        }
