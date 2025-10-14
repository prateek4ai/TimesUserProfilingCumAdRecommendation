"""Feature engineering utilities"""

import numpy as np
from typing import Dict, Any


class FeatureEngineer:
    """Feature engineering for CTR prediction."""
    
    def __init__(self):
        self.categorical_features = [
            'category_l1', 'device_type', 'hour', 
            'day_of_week', 'month', 'is_sponsored', 'exposure_bucket'
        ]
        self.numerical_features = [
            'position', 'user_ctr', 'item_ctr'
        ]
    
    def extract_features(self, user_id: int, item_id: int, **kwargs) -> np.ndarray:
        """
        Extract features for prediction.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            **kwargs: Additional features
            
        Returns:
            Feature vector
        """
        # Categorical (7 features)
        categorical = np.array([
            hash(user_id) % 10,  # category_l1
            0,  # device_type
            kwargs.get('hour', 12),
            kwargs.get('day_of_week', 2),
            kwargs.get('month', 10),
            int(kwargs.get('is_sponsored', False)),
            hash(user_id) % 10  # exposure_bucket
        ])
        
        # Numerical (25 features)
        numerical = np.array([
            kwargs.get('position', 0),
            kwargs.get('user_ctr', 0.1),
            kwargs.get('item_ctr', 0.2),
            *([0.1] * 22)  # Additional features
        ])
        
        return np.concatenate([categorical, numerical])
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize numerical features."""
        numerical_part = features[7:]  # Skip categorical
        normalized = (numerical_part - numerical_part.mean()) / (numerical_part.std() + 1e-8)
        return np.concatenate([features[:7], normalized])
