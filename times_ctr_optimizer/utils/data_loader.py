"""Data loading utilities"""

import polars as pl
from pathlib import Path
from typing import Union


class DataLoader:
    """Load training and inference data."""
    
    def __init__(self, data_dir: Union[str, Path] = "outputs"):
        self.data_dir = Path(data_dir)
    
    def load_training_data(self) -> pl.DataFrame:
        """Load training dataset."""
        path = self.data_dir / "training_data.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")
        return pl.read_parquet(path)
    
    def load_user_features(self) -> pl.DataFrame:
        """Load user feature store."""
        path = self.data_dir / "user_feature_store.parquet"
        if not path.exists():
            raise FileNotFoundError(f"User features not found: {path}")
        return pl.read_parquet(path)
    
    def load_item_features(self) -> pl.DataFrame:
        """Load item feature store."""
        path = self.data_dir / "item_feature_store.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Item features not found: {path}")
        return pl.read_parquet(path)
    
    def load_all(self) -> dict:
        """Load all datasets."""
        return {
            "training": self.load_training_data(),
            "users": self.load_user_features(),
            "items": self.load_item_features()
        }
