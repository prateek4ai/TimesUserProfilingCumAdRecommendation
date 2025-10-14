"""Wide & Deep model architecture"""

import torch
import torch.nn as nn


class WideDeepModel(nn.Module):
    """Wide & Deep architecture for CTR prediction."""
    
    def __init__(self, categorical_dims, num_numerical, embedding_dim=8, deep_layers=None):
        super().__init__()
        
        if deep_layers is None:
            deep_layers = [64, 32, 16]
        
        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) 
            for dim in categorical_dims.values()
        ])
        
        # Deep network
        input_dim = len(categorical_dims) * embedding_dim + num_numerical
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in deep_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.deep = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass."""
        cat_feats = x[:, :7].long()
        num_feats = x[:, 7:]
        
        embeddings = [emb(cat_feats[:, i]) for i, emb in enumerate(self.embeddings)]
        deep_input = torch.cat(embeddings + [num_feats], dim=1)
        
        deep_out = self.deep(deep_input)
        logits = self.output(deep_out)
        return self.sigmoid(logits)
