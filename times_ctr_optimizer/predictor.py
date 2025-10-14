"""Main CTR prediction interface"""

import torch
import numpy as np
from typing import Dict, List, Union
from .models.wide_deep import WideDeepModel


class CTRPredictor:
    """
    Enterprise CTR prediction engine for Times Network.
    
    Example:
        >>> from times_ctr_optimizer import CTRPredictor
        >>> predictor = CTRPredictor(model_path="model.pt")
        >>> ctr = predictor.predict(user_id=123, item_id=456)
        >>> print(f"Predicted CTR: {ctr:.2%}")
    """
    
    def __init__(self, model_path: str = "outputs/best_wide_deep_model.pt"):
        """Initialize predictor with trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
    def _load_model(self, path: str) -> WideDeepModel:
        """Load trained model from disk."""
        categorical_dims = {
            'category_l1': 10, 'device_type': 3, 'hour': 24,
            'day_of_week': 7, 'month': 12, 'is_sponsored': 2,
            'exposure_bucket': 10
        }
        model = WideDeepModel(categorical_dims, num_numerical=25)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, user_id: int, item_id: int, **kwargs) -> float:
        """
        Predict CTR for user-item pair.
        
        Args:
            user_id: User identifier
            item_id: Item identifier
            **kwargs: Additional features
            
        Returns:
            Predicted CTR (0.0 to 1.0)
        """
        features = self._prepare_features(user_id, item_id, **kwargs)
        
        with torch.no_grad():
            tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            prediction = self.model(tensor).item()
        
        return float(prediction)
    
    def predict_batch(self, pairs: List[Dict]) -> List[float]:
        """
        Batch prediction for multiple user-item pairs.
        
        Args:
            pairs: List of {"user_id": int, "item_id": int} dicts
            
        Returns:
            List of predicted CTRs
        """
        return [self.predict(**pair) for pair in pairs]
    
    def _prepare_features(self, user_id: int, item_id: int, **kwargs) -> np.ndarray:
        """Prepare feature vector for model."""
        # Categorical features (7)
        categorical = np.array([
            hash(user_id) % 10,  # category_l1
            0,  # device_type
            kwargs.get('hour', 12),
            kwargs.get('day_of_week', 2),
            kwargs.get('month', 10),
            int(kwargs.get('is_sponsored', False)),
            hash(user_id) % 10  # exposure_bucket
        ])
        
        # Numerical features (25)
        numerical = np.array([
            kwargs.get('position', 0),
            kwargs.get('user_ctr', 0.1),
            kwargs.get('item_ctr', 0.2),
            *([0.1] * 22)  # Additional features
        ])
        
        return np.concatenate([categorical, numerical])
