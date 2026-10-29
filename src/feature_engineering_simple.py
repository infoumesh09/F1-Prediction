"""
Simplified Feature Engineering Module for F1 Championship Prediction
Works with pre-calculated championship statistics
Updated for 2026 season
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """Engineer features for F1 championship prediction from championship data"""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from championship data for machine learning
        
        Args:
            data: DataFrame with championship statistics
            
        Returns:
            DataFrame with features for ML model
        """
        logger.info("Creating features for F1 championship prediction...")
        
        # Start with the existing data
        features_df = data.copy()
        
        # Create target variable (is_champion: 1 if final_position == 1, 0 otherwise)
        features_df['is_champion'] = (features_df['final_position'] == 1).astype(int)
        
        # Performance features
        features_df['win_rate'] = features_df['wins'] / features_df['races_started']
        features_df['points_per_race'] = features_df['final_points'] / features_df['races_started']
        features_df['podiums_per_race'] = features_df['podiums'] / features_df['races_started']
        features_df['fastest_laps_rate'] = features_df['fastest_laps'] / features_df['races_started']
        
        # Relative performance features
        features_df['points_vs_constructor'] = features_df['final_points'] / features_df['constructor_final_points']
        features_df['wins_vs_constructor'] = features_df['wins'] / (features_df['constructor_wins'] + 1)  # +1 to avoid division by zero
        
        # Position-based features
        features_df['avg_position_improvement'] = features_df['avg_grid_position'] - features_df['avg_finish_position']
        features_df['position_consistency'] = 1 / (features_df['avg_finish_position'] + 1)  # Lower is better
        
        # Reliability features
        features_df['reliability_score'] = features_df['completion_rate']
        features_df['dnf_penalty'] = features_df['dnf_rate'] * -1  # Negative penalty for DNFs
        
        # Team strength features
        features_df['team_strength'] = features_df['constructor_final_points'] / 400  # Normalize team points
        features_df['team_competitiveness'] = (features_df['constructor_final_position'] <= 3).astype(int)
        
        # Historical performance features (simple version)
        # Group by driver and create lagged features
        for year in sorted(features_df['year'].unique())[1:]:  # Skip first year
            prev_year = year - 1
            prev_data = features_df[features_df['year'] == prev_year]
            
            if len(prev_data) > 0:
                # Create a mapping of driver to previous year performance
                prev_performance = prev_data.set_index('driver_id')[['final_position', 'final_points', 'wins']].to_dict('index')
                
                # Add previous year features for current year
                current_year_mask = features_df['year'] == year
                for idx, row in features_df[current_year_mask].iterrows():
                    driver_id = row['driver_id']
                    if driver_id in prev_performance:
                        features_df.loc[idx, 'prev_year_position'] = prev_performance[driver_id]['final_position']
                        features_df.loc[idx, 'prev_year_points'] = prev_performance[driver_id]['final_points']
                        features_df.loc[idx, 'prev_year_wins'] = prev_performance[driver_id]['wins']
                    else:
                        # New driver - set defaults
                        features_df.loc[idx, 'prev_year_position'] = 15  # Assume mid-field
                        features_df.loc[idx, 'prev_year_points'] = 50   # Assume moderate points
                        features_df.loc[idx, 'prev_year_wins'] = 0      # Assume no wins
        
        # Fill NaN values for first year or missing data
        features_df['prev_year_position'] = features_df['prev_year_position'].fillna(15)
        features_df['prev_year_points'] = features_df['prev_year_points'].fillna(50)
        features_df['prev_year_wins'] = features_df['prev_year_wins'].fillna(0)
        
        # Experience features (count of years in data)
        driver_experience = features_df.groupby('driver_id')['year'].transform('count')
        features_df['driver_experience'] = driver_experience
        
        # Store feature columns for later use
        # final_position defines the target (the champion is P1), so including
        # it would leak the answer into both validation and prediction.
        self.feature_columns = [col for col in features_df.columns if col not in
                               ['year', 'driver_id', 'driver_code', 'driver_name', 'nationality',
                                'constructor_id', 'constructor_name', 'final_position',
                                'is_champion']]
        
        logger.info(f"Created {len(self.feature_columns)} features for ML model")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns for ML model"""
        return self.feature_columns
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> tuple:
        """
        Prepare training data for ML model (exclude 2026 data)
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Filter out 2026 data for training (since it's prediction data)
        training_df = features_df[features_df['year'] < 2026].copy()
        
        # Remove rows with missing target
        clean_df = training_df.dropna(subset=['is_champion'])
        
        # Get features and target
        X = clean_df[self.feature_columns]
        y = clean_df['is_champion']
        
        # Fill any remaining NaN values
        X = X.fillna(X.median())
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features (excluding 2026 data)")
        
        return X, y, self.feature_columns
    
    def prepare_prediction_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare prediction data for ML model (2026 data only)
        
        Args:
            features_df: DataFrame with features (should include 2026 data)
            
        Returns:
            DataFrame ready for prediction
        """
        # Filter to only 2026 data for prediction
        prediction_df = features_df[features_df['year'] == 2026].copy()
        
        # Get features
        X = prediction_df[self.feature_columns]
        
        # Fill any missing values with median from training data (years < 2026)
        training_df = features_df[features_df['year'] < 2026]
        training_medians = training_df[self.feature_columns].median()
        X = X.fillna(training_medians)
        
        logger.info(f"Prediction data prepared: {X.shape[0]} samples, {X.shape[1]} features (2026 data only)")
        
        return X
