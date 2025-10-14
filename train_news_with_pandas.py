#!/usr/bin/env python3
"""News Pre-training with Pandas (handles malformed CSV better)"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests, zipfile, json, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from datetime import datetime

print("="*80)
print("🔬 UCI NEWS PRE-TRAINING (Pandas Version)")
print("👨‍🎓 Prateek | IIT Patna MTech AI | Times Network")
print("="*80)

os.makedirs('outputs', exist_ok=True)

class CTRDataset(Dataset):
    def __init__(self, features, labels, weights):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx], self.weights[idx]

class WideDeepModel(nn.Module):
    def __init__(self, categorical_dims, num_numerical, embedding_dim=8):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, embedding_dim) for dim in categorical_dims.values()])
        total_embed_dim = len(categorical_dims) * embedding_dim
        self.deep = nn.Sequential(nn.Linear(total_embed_dim + num_numerical, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU())
        self.wide = nn.Linear(num_numerical, 1)
        self.output = nn.Linear(33, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        cat_features = x[:, :len(self.embeddings)].long()
        num_features = x[:, len(self.embeddings):]
        embeds = [emb(cat_features[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(embeds, dim=1)
        deep_out = self.deep(torch.cat([cat_embeds, num_features], dim=1))
        wide_out = self.wide(num_features)
        return self.sigmoid(self.output(torch.cat([deep_out, wide_out], dim=1)))

# Download
print("\n📥 Downloading UCI News Aggregator...")
data_dir = Path("outputs")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
data_path = data_dir / "newsCorpora.csv"
if not data_path.exists():
    r = requests.get(url, stream=True, timeout=60)
    zip_path = data_dir / "news.zip"
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(8192): f.write(chunk)
    zipfile.ZipFile(zip_path).extract("newsCorpora.csv", str(data_dir))
    print("✅ Downloaded")
else:
    print("✅ Already available")

# Load with Pandas (handles malformed rows better)
print("\n🔧 Loading with Pandas...")
df = pd.read_csv(
    data_path, 
    sep='\t', 
    header=None,
    names=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"],
    on_bad_lines='skip',  # Skip malformed rows
    nrows=100000
)
print(f"✅ Loaded {len(df):,} articles")

# Process
np.random.seed(42)
category_map = {'b': 'Business', 't': 'SciTech', 'e': 'Entertainment', 'm': 'Health'}
df['category_l1'] = df['CATEGORY'].map(category_map).fillna('Business')
df['item_id'] = range(len(df))
df['user_id'] = df.index % 10000
base_ctr, boosts = 0.08, {'Entertainment': 0.05, 'SciTech': 0.03, 'Business': 0.02, 'Health': 0.04}
df['clicked'] = [np.random.binomial(1, base_ctr + boosts.get(cat, 0.0)) for cat in df['category_l1']]
df['position'] = df.index % 10
df['hour'] = df.index % 24
df['day_of_week'] = df.index % 7
df['month'] = 10
df['price'] = 0.0
df['device_type'] = 'mobile'

# Compute CTRs
user_ctr = df.groupby('user_id')['clicked'].mean().rename('user_ctr')
item_ctr = df.groupby('item_id')['clicked'].mean().rename('item_ctr')
df = df.merge(user_ctr, on='user_id').merge(item_ctr, on='item_id')
print(f"✅ CTR: {df['clicked'].mean():.3f}")

# Encode
print("\n🏗️ Encoding features...")
categorical_features = ['category_l1', 'device_type']
numerical_features = ['position', 'hour', 'day_of_week', 'month', 'price', 'user_ctr', 'item_ctr']
label_encoders = {}
cat_encoded = {}
for col in categorical_features:
    le = LabelEncoder()
    cat_encoded[col] = le.fit_transform(df[col].fillna('unknown').astype(str))
    label_encoders[col] = le

numerical_data = df[numerical_features].fillna(0).values
cat_array = np.column_stack([cat_encoded[col] for col in categorical_features])
features = np.hstack([cat_array, numerical_data])
labels = df['clicked'].values
weights = np.ones(len(labels))
print(f"✅ Features: {features.shape}")

# Train
print("\n🚀 Training...")
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(features, labels, weights, test_size=0.2, random_state=42, stratify=labels)
train_dataset = CTRDataset(X_train, y_train, w_train)
val_dataset = CTRDataset(X_val, y_val, w_val)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)

categorical_dims = {col: len(label_encoders[col].classes_) for col in categorical_features}
model = WideDeepModel(categorical_dims, len(numerical_features))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

best_auc = 0
for epoch in range(5):
    model.train()
    for feat, lab, _ in train_loader:
        feat, lab = feat.to(device), lab.to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(feat), lab)
        loss.backward()
        optimizer.step()
    
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for feat, lab, _ in val_loader:
            preds.extend(model(feat.to(device)).cpu().numpy())
            labs.extend(lab.numpy())
    auc = roc_auc_score(labs, preds)
    best_auc = max(best_auc, auc)
    print(f"Epoch {epoch+1}/5: Val AUC={auc:.4f}")

# Save
print("\n💾 Saving...")
model_path = data_dir / 'news_pretrained_model.pth'
torch.save(model.state_dict(), model_path)
metadata = {"model_type": "WideDeepModel", "dataset": "UCI News Aggregator", "best_auc": float(best_auc),
            "training_samples": len(X_train), "num_articles": len(df), "categories": list(category_map.values()),
            "developer": "Prateek (IIT Patna MTech AI | Times Network)", "date": datetime.now().isoformat()}
json.dump(metadata, open(data_dir / 'news_model_metadata.json', 'w'), indent=2)

print("="*80)
print(f"🎊 COMPLETE! Best AUC: {best_auc:.4f}")
print(f"💾 {model_path}")
print(f"📊 {len(df):,} articles, CTR: {df['clicked'].mean():.3f}")
print("="*80)
