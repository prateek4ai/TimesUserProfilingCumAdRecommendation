"""Evaluation metrics"""

import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate AUC score.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        AUC score
    """
    return roc_auc_score(y_true, y_pred)


def calculate_ctr(clicks: np.ndarray, impressions: np.ndarray) -> float:
    """
    Calculate click-through rate.
    
    Args:
        clicks: Number of clicks
        impressions: Number of impressions
        
    Returns:
        CTR
    """
    return np.sum(clicks) / np.sum(impressions) if np.sum(impressions) > 0 else 0.0


def calculate_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Calculate Precision@K.
    
    Args:
        y_true: True labels
        y_pred: Predicted scores
        k: Number of top items
        
    Returns:
        Precision@K
    """
    top_k_idx = np.argsort(y_pred)[-k:]
    return np.sum(y_true[top_k_idx]) / k


def calculate_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Calculate NDCG@K.
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        k: Number of top items
        
    Returns:
        NDCG@K
    """
    def dcg(scores):
        return np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2)))
    
    top_k_idx = np.argsort(y_pred)[-k:][::-1]
    dcg_score = dcg(y_true[top_k_idx])
    
    ideal_idx = np.argsort(y_true)[-k:][::-1]
    idcg_score = dcg(y_true[ideal_idx])
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0
