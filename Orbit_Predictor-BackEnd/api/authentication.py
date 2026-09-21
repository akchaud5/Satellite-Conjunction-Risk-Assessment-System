# api/authentication.py

import jwt
from django.conf import settings
from rest_framework import authentication, exceptions
from .models import User


class JWTAuthentication(authentication.BaseAuthentication):
    """
    Custom JWT Authentication class.
    """

    def authenticate(self, request):
        auth_header = authentication.get_authorization_header(request)

        if not auth_header:
            return None  # No authentication credentials provided

        try:
            prefix, token = auth_header.decode('utf-8').split(' ')
            if prefix.lower() != 'bearer':
                return None  # Invalid prefix
        except ValueError:
            raise exceptions.AuthenticationFailed('Invalid token header. No credentials provided.')

        return self.authenticate_credentials(token)

    def authenticate_header(self, request):
        """Return a WWW-Authenticate value so DRF answers 401, not 403.

        Without this DRF has no challenge to send and downgrades every
        authentication failure to 403. The frontend keys its "session expired,
        clear the token and bounce to /login" handling off 401 in ten places,
        so an expired token left the user staring at a dead page with a stale
        token still in localStorage.
        """
        return 'Bearer realm="api"'

    def authenticate_credentials(self, token):
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('Token has expired.')
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token.')

        # Refresh tokens live seven days and exist only to mint access tokens.
        # Without this check they were accepted here as bearer credentials,
        # making every session's effective access lifetime seven days.
        if payload.get('token_type') != 'access':
            raise exceptions.AuthenticationFailed(
                'Invalid token: an access token is required.'
            )

        user_id = payload.get('user_id')
        role = payload.get('role')

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found.')

        if not user.is_active:
            raise exceptions.AuthenticationFailed('User account is disabled.')

        # Optionally, you can verify the role matches
        if user.role != role:
            raise exceptions.AuthenticationFailed('Invalid token payload.')

        return (user, token)
