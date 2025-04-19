"""
Management command to create a test user.
"""

from django.core.management.base import BaseCommand
from api.models.user import User

class Command(BaseCommand):
    help = 'Create a test user for development purposes'

    def add_arguments(self, parser):
        parser.add_argument('--email', type=str, default='test@example.com', help='User email')
        parser.add_argument('--password', type=str, default='password123', help='User password')
        parser.add_argument('--role', type=str, default='admin', help='User role: admin, collision_analyst, or user')

    def handle(self, *args, **options):
        email = options['email']
        password = options['password']
        role = options['role']

        # Check if user already exists
        if User.objects.filter(email=email).exists():
            self.stdout.write(self.style.WARNING(f'User with email {email} already exists.'))
            user = User.objects.get(email=email)
            user.set_password(password)
            user.role = role
            user.save()
            self.stdout.write(self.style.SUCCESS(f'Updated existing user {email} with new password and role {role}.'))
        else:
            # Create new user
            user = User.objects.create_user(
                email=email,
                password=password,
                role=role,
                is_staff=role == 'admin',
                is_superuser=role == 'admin'
            )
            self.stdout.write(self.style.SUCCESS(f'Successfully created user {email} with role {role}.'))