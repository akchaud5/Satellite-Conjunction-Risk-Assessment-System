"""
Feature engineering module for satellite collision prediction ML models.

This module handles the creation and transformation of features
from the raw CDM data for machine learning models.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from ..models.cdm import CDM


def extract_features_from_cdm(cdm, include_derived=True):
    """
    Extract features from a CDM object for machine learning models.
    
    Args:
        cdm: CDM model instance
        include_derived: Whether to include derived/calculated features
        
    Returns:
        Dictionary of features
    """
    # Base features from the CDM
    features = {
        # Relative position and velocity
        'rel_pos_x': cdm.sat2_x - cdm.sat1_x,
        'rel_pos_y': cdm.sat2_y - cdm.sat1_y,
        'rel_pos_z': cdm.sat2_z - cdm.sat1_z,
        'rel_vel_x': cdm.sat2_x_dot - cdm.sat1_x_dot,
        'rel_vel_y': cdm.sat2_y_dot - cdm.sat1_y_dot, 
        'rel_vel_z': cdm.sat2_z_dot - cdm.sat1_z_dot,
        
        # Individual satellite positions and velocities
        'sat1_x': cdm.sat1_x,
        'sat1_y': cdm.sat1_y,
        'sat1_z': cdm.sat1_z,
        'sat1_vx': cdm.sat1_x_dot,
        'sat1_vy': cdm.sat1_y_dot,
        'sat1_vz': cdm.sat1_z_dot,
        
        'sat2_x': cdm.sat2_x,
        'sat2_y': cdm.sat2_y,
        'sat2_z': cdm.sat2_z,
        'sat2_vx': cdm.sat2_x_dot,
        'sat2_vy': cdm.sat2_y_dot,
        'sat2_vz': cdm.sat2_z_dot,
        
        # Covariance matrix elements
        'sat1_cov_rr': cdm.sat1_cov_rr,
        'sat1_cov_rt': cdm.sat1_cov_rt,
        'sat1_cov_rn': cdm.sat1_cov_rn,
        'sat1_cov_tt': cdm.sat1_cov_tt,
        'sat1_cov_tn': cdm.sat1_cov_tn,
        'sat1_cov_nn': cdm.sat1_cov_nn,
        
        'sat2_cov_rr': cdm.sat2_cov_rr,
        'sat2_cov_rt': cdm.sat2_cov_rt,
        'sat2_cov_rn': cdm.sat2_cov_rn,
        'sat2_cov_tt': cdm.sat2_cov_tt,
        'sat2_cov_tn': cdm.sat2_cov_tn,
        'sat2_cov_nn': cdm.sat2_cov_nn,
        
        # Original miss distance from CDM
        'miss_distance': cdm.miss_distance,
        
        # Maneuverability flags
        'sat1_maneuverable': 1 if cdm.sat1_maneuverable.lower() == 'yes' else 0,
        'sat2_maneuverable': 1 if cdm.sat2_maneuverable.lower() == 'yes' else 0,
    }
    
    # Derived features
    if include_derived:
        # Relative position magnitude
        rel_pos = np.array([features['rel_pos_x'], features['rel_pos_y'], features['rel_pos_z']])
        rel_vel = np.array([features['rel_vel_x'], features['rel_vel_y'], features['rel_vel_z']])
        
        # Compute magnitudes
        features['rel_pos_mag'] = np.linalg.norm(rel_pos)
        features['rel_vel_mag'] = np.linalg.norm(rel_vel)
        
        # Compute angles
        if features['rel_pos_mag'] > 0 and features['rel_vel_mag'] > 0:
            features['rel_angle'] = np.arccos(
                np.clip(np.dot(rel_pos, rel_vel) / (features['rel_pos_mag'] * features['rel_vel_mag']), -1.0, 1.0)
            )
        else:
            features['rel_angle'] = 0
            
        # Compute combined covariance determinant (uncertainty volume)
        sat1_cov = np.array([
            [features['sat1_cov_rr'], features['sat1_cov_rt'], features['sat1_cov_rn']],
            [features['sat1_cov_rt'], features['sat1_cov_tt'], features['sat1_cov_tn']],
            [features['sat1_cov_rn'], features['sat1_cov_tn'], features['sat1_cov_nn']]
        ])
        
        sat2_cov = np.array([
            [features['sat2_cov_rr'], features['sat2_cov_rt'], features['sat2_cov_rn']],
            [features['sat2_cov_rt'], features['sat2_cov_tt'], features['sat2_cov_tn']],
            [features['sat2_cov_rn'], features['sat2_cov_tn'], features['sat2_cov_nn']]
        ])
        
        # Covariance properties (handle potential numerical issues)
        try:
            features['sat1_cov_det'] = max(np.linalg.det(sat1_cov), 1e-10)
            features['sat2_cov_det'] = max(np.linalg.det(sat2_cov), 1e-10)
            features['combined_cov_det'] = max(np.linalg.det(sat1_cov + sat2_cov), 1e-10)
        except:
            # Fallback values in case of singular matrices
            features['sat1_cov_det'] = 1e-10
            features['sat2_cov_det'] = 1e-10  
            features['combined_cov_det'] = 1e-10
        
        # Time of closest approach features
        if hasattr(cdm, 'tca') and cdm.tca:
            # Extract time components as features
            features['tca_hour'] = cdm.tca.hour
            features['tca_day'] = cdm.tca.day
            
            # Compute velocity ratio (relative to escape velocity)
            # This is a simplified approximation
            sat1_pos = np.array([features['sat1_x'], features['sat1_y'], features['sat1_z']])
            sat1_vel = np.array([features['sat1_vx'], features['sat1_vy'], features['sat1_vz']])
            sat1_pos_mag = np.linalg.norm(sat1_pos)
            sat1_vel_mag = np.linalg.norm(sat1_vel)
            
            # Earth's gravitational parameter (km^3/s^2)
            mu_earth = 398600.4418
            
            # Calculate escape velocity at this position (km/s)
            if sat1_pos_mag > 0:
                v_escape = np.sqrt(2 * mu_earth / sat1_pos_mag)
                features['vel_escape_ratio'] = sat1_vel_mag / v_escape
            else:
                features['vel_escape_ratio'] = 0
                
    return features


def prepare_dataset_from_cdms(cdms, target_variable=None):
    """
    Prepare a dataset from a list of CDM objects.
    
    Args:
        cdms: List of CDM objects
        target_variable: Name of the target variable (if None, only features are returned)
        
    Returns:
        pandas DataFrame with features (and target if specified)
    """
    features_list = []
    target_values = []
    
    for cdm in cdms:
        features = extract_features_from_cdm(cdm)
        features['cdm_id'] = str(cdm.id)
        features_list.append(features)
        
        if target_variable:
            if target_variable == 'probability_of_collision':
                # Get the latest collision probability value
                collisions = cdm.collisions.all().order_by('-created_at')
                if collisions.exists():
                    target_values.append(collisions.first().probability_of_collision)
                else:
                    target_values.append(None)
            elif target_variable == 'miss_distance':
                target_values.append(cdm.miss_distance)
            else:
                # Handle other target variables if needed
                target_values.append(None)
    
    df = pd.DataFrame(features_list)
    
    if target_variable and any(v is not None for v in target_values):
        df[target_variable] = target_values
        
    return df


def engineer_advanced_features(df):
    """
    Add advanced/derived features to a dataset.
    
    Args:
        df: pandas DataFrame with basic features
        
    Returns:
        DataFrame with additional engineered features
    """
    # Make a copy to avoid modifying the original
    df_eng = df.copy()
    
    # Relative position and velocity magnitudes
    if all(col in df.columns for col in ['rel_pos_x', 'rel_pos_y', 'rel_pos_z']):
        df_eng['rel_pos_mag'] = np.sqrt(
            df_eng['rel_pos_x']**2 + 
            df_eng['rel_pos_y']**2 + 
            df_eng['rel_pos_z']**2
        )
        
    if all(col in df.columns for col in ['rel_vel_x', 'rel_vel_y', 'rel_vel_z']):
        df_eng['rel_vel_mag'] = np.sqrt(
            df_eng['rel_vel_x']**2 + 
            df_eng['rel_vel_y']**2 + 
            df_eng['rel_vel_z']**2
        )
        
    # Combined covariance volume (uncertainty)
    if all(col in df.columns for col in ['sat1_cov_rr', 'sat2_cov_rr']):
        df_eng['combined_pos_uncertainty'] = np.sqrt(
            df_eng['sat1_cov_rr'] + df_eng['sat2_cov_rr']
        )
        
    # Normalized miss distance (by combined position uncertainty)
    if 'combined_pos_uncertainty' in df_eng.columns and 'miss_distance' in df_eng.columns:
        # Avoid division by zero
        df_eng['norm_miss_distance'] = df_eng['miss_distance'] / df_eng['combined_pos_uncertainty'].replace(0, 1e-10)
    
    # Time-to-closest approach in hours (if time features available)
    if 'tca' in df.columns and hasattr(df['tca'].iloc[0], 'timestamp'):
        now = pd.Timestamp.now(df['tca'].iloc[0].tzinfo)
        df_eng['hours_to_tca'] = (df_eng['tca'] - now).dt.total_seconds() / 3600
    
    return df_eng


def normalize_features(train_df, test_df=None, exclude_cols=None):
    """
    Normalize features using min-max scaling.
    
    Args:
        train_df: Training data DataFrame
        test_df: Optional test data to normalize with training stats
        exclude_cols: Columns to exclude from normalization
        
    Returns:
        Normalized DataFrames (train and test if provided)
    """
    if exclude_cols is None:
        exclude_cols = []
        
    # Identify numerical columns to normalize
    num_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    norm_cols = [col for col in num_cols if col not in exclude_cols]
    
    # Create copies to avoid modifying originals
    train_norm = train_df.copy()
    
    # Calculate min and max for each column
    min_vals = train_df[norm_cols].min()
    max_vals = train_df[norm_cols].max()
    
    # Apply normalization
    for col in norm_cols:
        # Handle columns with no variation
        if max_vals[col] > min_vals[col]:
            train_norm[col] = (train_df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
        else:
            train_norm[col] = 0.5  # Set to constant value if no variation
            
    if test_df is not None:
        test_norm = test_df.copy()
        for col in norm_cols:
            if max_vals[col] > min_vals[col]:
                test_norm[col] = (test_df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])
                # Clip values to [0,1] range for test data
                test_norm[col] = np.clip(test_norm[col], 0, 1)
            else:
                test_norm[col] = 0.5
        return train_norm, test_norm, min_vals, max_vals
        
    return train_norm, min_vals, max_vals