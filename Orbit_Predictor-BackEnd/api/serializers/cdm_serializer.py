from rest_framework import serializers
from ..models import CDM


class CDMSerializer(serializers.ModelSerializer):
    """The single CDM serializer.

    There used to be two classes with this name -- this one (``fields =
    '__all__'``) and another in user_serializer.py with an explicit field list.
    Both were exported from serializers/__init__.py, so which one callers got
    depended purely on import order; the explicit-list one won and silently
    dropped the fourteen object-metadata fields below from every CDM response.
    CDMCreateView stores them and the CDM detail page renders them, so those
    rows had always come back blank in the UI.

    Fields are listed explicitly rather than using '__all__' so that adding a
    column to the model is never enough, on its own, to publish it over the API.
    """

    class Meta:
        model = CDM
        fields = [
            'id',
            'ccsds_cdm_version',
            'creation_date',
            'originator',
            'message_id',
            'privacy',
            'tca',
            'miss_distance',
            # Satellite 1
            'sat1_object',
            'sat1_object_designator',
            'sat1_maneuverable',
            'sat1_x',
            'sat1_y',
            'sat1_z',
            'sat1_x_dot',
            'sat1_y_dot',
            'sat1_z_dot',
            'sat1_cov_rr',
            'sat1_cov_rt',
            'sat1_cov_rn',
            'sat1_cov_tr',
            'sat1_cov_tt',
            'sat1_cov_tn',
            'sat1_cov_nr',
            'sat1_cov_nt',
            'sat1_cov_nn',
            'sat1_catalog_name',
            'sat1_object_name',
            'sat1_international_designator',
            'sat1_object_type',
            'sat1_operator_organization',
            'sat1_covariance_method',
            'sat1_reference_frame',
            # Satellite 2
            'sat2_object',
            'sat2_object_designator',
            'sat2_maneuverable',
            'sat2_x',
            'sat2_y',
            'sat2_z',
            'sat2_x_dot',
            'sat2_y_dot',
            'sat2_z_dot',
            'sat2_cov_rr',
            'sat2_cov_rt',
            'sat2_cov_rn',
            'sat2_cov_tr',
            'sat2_cov_tt',
            'sat2_cov_tn',
            'sat2_cov_nr',
            'sat2_cov_nt',
            'sat2_cov_nn',
            'sat2_catalog_name',
            'sat2_object_name',
            'sat2_international_designator',
            'sat2_object_type',
            'sat2_operator_organization',
            'sat2_covariance_method',
            'sat2_reference_frame',
            'hard_body_radius',
        ]
        read_only_fields = ['id']
