"""Regression tests for the API.

These cover the defects fixed alongside them, so that the specific things that
were broken stay fixed:

* role escalation through the self-service user update path
* refresh tokens being accepted as access tokens
* CDM responses silently dropping object-metadata fields
* pickle loads escaping the managed model directory
* the app importing without MATLAB installed
"""

import datetime
import uuid

import jwt
from django.conf import settings
from django.core.exceptions import SuspiciousFileOperation
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from api.models import CDM, User
from api.models.ml_model import MLModel


def make_user(email="pilot@example.com", password="correct-horse", role="user"):
    user = User.objects.create_user(email=email, password=password)
    if user.role != role:
        user.role = role
        user.save()
    return user


def make_cdm(**overrides):
    """A CDM with every required numeric field populated."""
    fields = {
        "ccsds_cdm_version": "1.0",
        "creation_date": "2025-01-01T00:00:00Z",
        "originator": "TEST",
        "message_id": f"MSG-{uuid.uuid4()}",
        "tca": "2025-01-02T00:00:00Z",
        "miss_distance": 1234.5,
        "privacy": True,
        "hard_body_radius": 20.0,
    }
    for sat in ("sat1", "sat2"):
        fields[f"{sat}_object"] = "PAYLOAD"
        fields[f"{sat}_object_designator"] = "25544"
        fields[f"{sat}_maneuverable"] = "YES"
        for axis in ("x", "y", "z"):
            fields[f"{sat}_{axis}"] = 1.0
            fields[f"{sat}_{axis}_dot"] = 0.1
        for a in ("r", "t", "n"):
            for b in ("r", "t", "n"):
                fields[f"{sat}_cov_{a}{b}"] = 1.0 if a == b else 0.0
    fields.update(overrides)
    return CDM.objects.create(**fields)


class AuthTokenTests(APITestCase):
    """Login, refresh, and the access/refresh token split."""

    def setUp(self):
        self.password = "correct-horse-battery"
        self.user = make_user(password=self.password)

    def login(self):
        response = self.client.post(
            reverse("login"),
            {"email": self.user.email, "password": self.password},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        return response.data

    def test_login_returns_access_and_refresh_tokens(self):
        data = self.login()
        self.assertIn("access", data)
        self.assertIn("refresh_token", data)

    def test_tokens_carry_distinct_token_types(self):
        data = self.login()
        access = jwt.decode(
            data["access"], settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        refresh = jwt.decode(
            data["refresh_token"], settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        self.assertEqual(access["token_type"], "access")
        self.assertEqual(refresh["token_type"], "refresh")

    def test_access_token_authenticates(self):
        data = self.login()
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {data['access']}")
        response = self.client.get(reverse("current_user"))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["email"], self.user.email)

    def test_refresh_token_is_rejected_as_a_bearer_credential(self):
        """A 7-day refresh token must not work as a 24-hour access token."""
        data = self.login()
        self.client.credentials(
            HTTP_AUTHORIZATION=f"Bearer {data['refresh_token']}"
        )
        response = self.client.get(reverse("current_user"))
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_access_token_is_rejected_by_the_refresh_endpoint(self):
        data = self.login()
        response = self.client.post(
            reverse("refresh_token"),
            {"refresh_token": data["access"]},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_refresh_returns_a_usable_access_token(self):
        data = self.login()
        response = self.client.post(
            reverse("refresh_token"),
            {"refresh_token": data["refresh_token"]},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.client.credentials(
            HTTP_AUTHORIZATION=f"Bearer {response.data['access']}"
        )
        self.assertEqual(
            self.client.get(reverse("current_user")).status_code,
            status.HTTP_200_OK,
        )

    def test_disabled_user_cannot_authenticate(self):
        data = self.login()
        self.user.is_active = False
        self.user.save()
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {data['access']}")
        self.assertEqual(
            self.client.get(reverse("current_user")).status_code,
            status.HTTP_401_UNAUTHORIZED,
        )

    def test_expired_token_is_rejected(self):
        past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        token = jwt.encode(
            {
                "user_id": str(self.user.id),
                "role": self.user.role,
                "token_type": "access",
                "exp": past,
                "iat": past - datetime.timedelta(hours=1),
            },
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {token}")
        self.assertEqual(
            self.client.get(reverse("current_user")).status_code,
            status.HTTP_401_UNAUTHORIZED,
        )

    def test_login_with_wrong_password_fails(self):
        response = self.client.post(
            reverse("login"),
            {"email": self.user.email, "password": "wrong"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class RolePrivilegeTests(APITestCase):
    """Role must not be settable through the self-service update path."""

    def setUp(self):
        self.password = "correct-horse-battery"
        self.user = make_user(password=self.password)
        self.client.force_authenticate(user=self.user)

    def test_user_cannot_promote_themselves_to_admin(self):
        response = self.client.patch(
            reverse("user-detail", args=[str(self.user.id)]),
            {"role": "admin"},
            format="json",
        )
        self.user.refresh_from_db()
        self.assertEqual(self.user.role, "user")
        self.assertNotEqual(response.data.get("role"), "admin")

    def test_user_cannot_promote_themselves_to_analyst(self):
        self.client.patch(
            reverse("user-detail", args=[str(self.user.id)]),
            {"role": "collision_analyst"},
            format="json",
        )
        self.user.refresh_from_db()
        self.assertEqual(self.user.role, "user")

    def test_user_cannot_edit_another_user(self):
        other = make_user(email="other@example.com")
        response = self.client.patch(
            reverse("user-detail", args=[str(other.id)]),
            {"email": "hijacked@example.com"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_user_can_still_update_their_own_password(self):
        response = self.client.patch(
            reverse("user-detail", args=[str(self.user.id)]),
            {"password": "a-brand-new-password"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("a-brand-new-password"))

    def test_admin_can_still_assign_roles(self):
        admin = make_user(email="admin@example.com", role="admin")
        target = make_user(email="target@example.com")
        self.client.force_authenticate(user=admin)
        response = self.client.patch(
            reverse("user-detail", args=[str(target.id)]),
            {"role": "collision_analyst"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        target.refresh_from_db()
        self.assertEqual(target.role, "collision_analyst")

    def test_admin_cannot_assign_an_unknown_role(self):
        admin = make_user(email="admin2@example.com", role="admin")
        target = make_user(email="target2@example.com")
        self.client.force_authenticate(user=admin)
        response = self.client.patch(
            reverse("user-detail", args=[str(target.id)]),
            {"role": "superuser"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class RegistrationRoleTests(APITestCase):
    """Role on registration is derived, never taken from the request."""

    def test_self_declared_role_is_ignored_on_register(self):
        response = self.client.post(
            reverse("register"),
            {
                "email": "nobody@example.com",
                "password": "correct-horse-battery",
                "role": "admin",
            },
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["role"], "user")

    def test_space_agency_domain_becomes_analyst(self):
        response = self.client.post(
            reverse("register"),
            {"email": "someone@nasa.gov", "password": "correct-horse-battery"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data["role"], "collision_analyst")

    def test_duplicate_email_is_rejected(self):
        make_user(email="dupe@example.com")
        response = self.client.post(
            reverse("register"),
            {"email": "dupe@example.com", "password": "correct-horse-battery"},
            format="json",
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class CDMSerializerFieldTests(APITestCase):
    """CDM responses must carry the object-metadata fields the UI renders."""

    METADATA_FIELDS = [
        "catalog_name",
        "object_name",
        "international_designator",
        "object_type",
        "operator_organization",
        "covariance_method",
        "reference_frame",
    ]

    def setUp(self):
        self.user = make_user(role="collision_analyst")
        self.client.force_authenticate(user=self.user)
        self.cdm = make_cdm(
            sat1_catalog_name="SATCAT",
            sat1_object_name="ISS (ZARYA)",
            sat1_international_designator="1998-067A",
            sat1_object_type="PAYLOAD",
            sat1_operator_organization="NASA",
            sat1_covariance_method="CALCULATED",
            sat1_reference_frame="ITRF",
        )

    def test_detail_response_includes_object_metadata(self):
        response = self.client.get(reverse("cdm-detail", args=[self.cdm.id]))
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        for sat in ("sat1", "sat2"):
            for field in self.METADATA_FIELDS:
                self.assertIn(f"{sat}_{field}", response.data)

    def test_detail_response_returns_the_stored_values(self):
        response = self.client.get(reverse("cdm-detail", args=[self.cdm.id]))
        self.assertEqual(response.data["sat1_object_name"], "ISS (ZARYA)")
        self.assertEqual(response.data["sat1_international_designator"], "1998-067A")

    def test_covariance_is_serialized(self):
        response = self.client.get(reverse("cdm-detail", args=[self.cdm.id]))
        self.assertEqual(response.data["sat1_cov_rr"], 1.0)
        self.assertEqual(response.data["sat1_cov_rt"], 0.0)


class CDMVisibilityTests(APITestCase):
    """Regular users only see CDMs flagged as shareable."""

    def setUp(self):
        self.public = make_cdm(privacy=True)
        self.private = make_cdm(privacy=False)

    def test_regular_user_sees_only_public_cdms(self):
        self.client.force_authenticate(user=make_user())
        ids = {row["id"] for row in self.client.get(reverse("cdm-list")).data}
        self.assertIn(self.public.id, ids)
        self.assertNotIn(self.private.id, ids)

    def test_analyst_sees_all_cdms(self):
        self.client.force_authenticate(
            user=make_user(email="analyst@example.com", role="collision_analyst")
        )
        ids = {row["id"] for row in self.client.get(reverse("cdm-list")).data}
        self.assertIn(self.public.id, ids)
        self.assertIn(self.private.id, ids)

    def test_anonymous_access_is_refused(self):
        response = self.client.get(reverse("cdm-list"))
        self.assertIn(
            response.status_code,
            (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN),
        )


class MLModelFilePathTests(APITestCase):
    """Model unpickling stays inside the managed directory."""

    def test_path_outside_the_models_dir_is_refused(self):
        model = MLModel.objects.create(
            name="evil", model_type="collision_probability",
            algorithm="random_forest", version="1.0.0",
            file_path="/etc/passwd",
        )
        # Only the basename is honoured, so this resolves inside ML_MODELS_DIR
        # and simply does not exist -- it never reads /etc/passwd.
        with self.assertRaises((FileNotFoundError, SuspiciousFileOperation)):
            model.resolved_model_path()

    def test_traversal_path_is_refused(self):
        model = MLModel.objects.create(
            name="evil2", model_type="collision_probability",
            algorithm="random_forest", version="1.0.0",
            file_path="../../../../etc/passwd",
        )
        with self.assertRaises((FileNotFoundError, SuspiciousFileOperation)):
            model.resolved_model_path()

    def test_missing_file_path_raises(self):
        model = MLModel.objects.create(
            name="empty", model_type="collision_probability",
            algorithm="random_forest", version="1.0.0",
        )
        with self.assertRaises(FileNotFoundError):
            model.resolved_model_path()

    def test_file_path_is_not_writable_through_the_api(self):
        admin = make_user(email="mladmin@example.com", role="admin")
        self.client.force_authenticate(user=admin)
        model = MLModel.objects.create(
            name="m", model_type="collision_probability",
            algorithm="random_forest", version="1.0.0",
            file_path="legit.pkl",
        )
        self.client.patch(
            reverse("ml-model-detail", args=[str(model.id)]),
            {"file_path": "/etc/passwd"},
            format="json",
        )
        model.refresh_from_db()
        self.assertEqual(model.file_path, "legit.pkl")


class MatlabOptionalTests(APITestCase):
    """The app must import and serve without a MATLAB installation."""

    def test_matlab_runtime_reports_unavailable_rather_than_raising(self):
        from api import matlab_runtime

        # Either answer is valid depending on the host; the point is that
        # asking does not blow up and does not start an engine.
        self.assertIn(matlab_runtime.matlab_available(), (True, False))

    def test_importing_models_does_not_require_matlab(self):
        import importlib

        # Would raise ImportError if matlab.engine were imported at module
        # scope, as it used to be.
        importlib.import_module("api.models.collision")
        importlib.import_module("api.views.tradespace_heatmap_views")
        importlib.import_module("api.views.tradespace_linear_views")
