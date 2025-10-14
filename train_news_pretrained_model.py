#!/usr/bin/env python3
"""
Complete Standalone News Pre-training Script
UCI News Aggregator Dataset Training
Developed by: Prateek (IIT Patna MTech AI | Times Network)
Email: prat.cann.170701@gmail.com
"""

# ALL IMPORTS
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import json
from datetime import datetime
import os

print("=" * 80)
print("🔬 PRE-TRAINING ON UCI NEWS AGGREGATOR DATASET")
print("🎓 IIT Patna MTech AI Project | 🏢 Times Network Application")
print("👨‍🎓 Developer: Prateek (prat.cann.170701@gmail.com)")
print("=" * 80)

# Create outputs directory
os.makedirs('outputs', exist_ok=True)

# Define Dataset class
class CTRDataset(Dataset):
    def __init__(self, features, labels, weights):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.weights = torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]

# Define WideDeepModel
class WideDeepModel(nn.Module):
    def __init__(self, categorical_dims, num_numerical, embedding_dim=8):
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embedding_dim) for dim in categorical_dims.values()
        ])
        
        total_embed_dim = len(categorical_dims) * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(total_embed_dim + num_numerical, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.wide = nn.Linear(num_numerical, 1)
        self.output = nn.Linear(33, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        cat_features = x[:, :len(self.embeddings)].long()
        num_features = x[:, len(self.embeddings):]
        
        embeds = [emb(cat_features[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeds = torch.cat(embeds, dim=1)
        
        deep_input = torch.cat([cat_embeds, num_features], dim=1)
        deep_out = self.deep(deep_input)
        wide_out = self.wide(num_features)
        
        combined = torch.cat([deep_out, wide_out], dim=1)
        output = self.output(combined)
        
        return self.sigmoid(output)

# Step 1: Download UCI News Dataset
print("\n📥 Step 1: Downloading UCI News Aggregator dataset...")
data_dir = Path("outputs")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
zip_path = data_dir / "NewsAggregatorDataset.zip"
data_path = data_dir / "newsCorpora.csv"

if not data_path.exists():
    print("   Downloading...")
    r = requests.get(url, stream=True, timeout=60)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("newsCorpora.csv", str(data_dir))
    print("✅ Dataset downloaded")
else:
    print("✅ Dataset already available")

# Step 2: Load and process
print("\n🔧 Step 2: Processing news dataset...")
news_df = pl.read_csv(
    str(data_path),
    separator="\t",
    has_header=False,
    ignore_errors=True,
    new_columns=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
)







n_records = min(len(news_df), 100000)  # Use 100k records
news_df = news_df.head(n_records)
print(f"   Processing {n_records:,} news articles")

# Map categories
category_map = {'b': 'Business', 't': 'SciTech', 'e': 'Entertainment', 'm': 'Health'}

# Simulate CTR
np.random.seed(42)
base_ctr = 0.08
category_boost = {'Entertainment': 0.05, 'SciTech': 0.03, 'Business': 0.02, 'Health': 0.04}

clicks = []
for cat in news_df['CATEGORY']:
    mapped_cat = category_map.get(cat, 'Business')
    ctr = base_ctr + category_boost.get(mapped_cat, 0.0)
    clicks.append(np.random.binomial(1, ctr))

# Create features
news_training = news_df.with_columns([
    pl.col("CATEGORY").replace(category_map).alias("category_l1"),
    pl.int_range(0, pl.len()).alias("item_id"),
    (pl.int_range(0, pl.len()) % 10000).alias("user_id"),
    pl.Series("clicked", clicks),
    (pl.int_range(0, pl.len()) % 10).alias("position"),
    (pl.int_range(0, pl.len()) % 24).alias("hour"),
    (pl.int_range(0, pl.len()) % 7).alias("day_of_week"),
    pl.lit(10).alias("month"),
    pl.lit(0.0).alias("price"),
    pl.lit("mobile").alias("device_type"),
])

# Compute CTRs
user_ctrs = news_training.group_by("user_id").agg([pl.col("clicked").mean().alias("user_ctr")])
item_ctrs = news_training.group_by("item_id").agg([pl.col("clicked").mean().alias("item_ctr")])

news_training = news_training.join(user_ctrs, on="user_id", how="left")
news_training = news_training.join(item_ctrs, on="item_id", how="left")

print(f"✅ Processed {len(news_training):,} records, CTR: {news_training['clicked'].mean():.3f}")

# Step 3: Prepare features
print("\n🏗️ Step 3: Preparing features...")
categorical_features = ['category_l1', 'device_type']
numerical_features = ['position', 'hour', 'day_of_week', 'month', 'price', 'user_ctr', 'item_ctr']

# Handle nulls in Polars
news_clean = news_training.with_columns([
    pl.col(c).fill_null("unknown").cast(pl.Utf8)
    for c in categorical_features if c in news_training.columns
])

df_pd = news_clean.to_pandas()

# Encode
label_encoders = {}
categorical_encoded = {}
for col in categorical_features:
    if col in df_pd.columns:
        le = LabelEncoder()
        categorical_encoded[col] = le.fit_transform(df_pd[col].astype(str))
        label_encoders[col] = le

numerical_data = df_pd[numerical_features].fillna(0).values
cat_array = np.column_stack([categorical_encoded[col] for col in categorical_features])
features = np.hstack([cat_array, numerical_data])
labels = df_pd['clicked'].values
weights = np.ones(len(labels))

print(f"✅ Features: {features.shape}, Positive: {labels.sum():,}")

# Step 4: Train
print("\n�� Step 4: Training CTR model...")
X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    features, labels, weights, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = CTRDataset(X_train, y_train, w_train)
val_dataset = CTRDataset(X_val, y_val, w_val)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

categorical_dims = {col: len(label_encoders[col].classes_) for col in categorical_features}
model = WideDeepModel(categorical_dims=categorical_dims, num_numerical=len(numerical_features))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

best_auc = 0
epochs = 5

print(f"\n🏋️  Training {epochs} epochs...")
for epoch in range(epochs):
    model.train()
    for features_batch, labels_batch, _ in train_loader:
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for features_batch, labels_batch, _ in val_loader:
            outputs = model(features_batch.to(device))
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels_batch.numpy())
    
    val_auc = roc_auc_score(val_labels, val_preds)
    if val_auc > best_auc:
        best_auc = val_auc
    
    print(f"Epoch {epoch+1}/{epochs}: Val AUC={val_auc:.4f}")

# Step 5: Save
print("\n💾 Step 5: Saving model...")
model_path = data_dir / 'news_pretrained_model.pth'
torch.save(model.state_dict(), model_path)

metadata = {
    "model_type": "WideDeepModel",
    "dataset": "UCI News Aggregator",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "best_auc": float(best_auc),
    "categories": list(category_map.values()),
    "overall_ctr": float(news_training['clicked'].mean()),
    "num_articles": len(news_training),
    "developer": "Prateek (IIT Patna MTech AI | Times Network)",
    "email": "prat.cann.170701@gmail.com",
    "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

metadata_path = data_dir / 'news_model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("=" * 80)
print("🎊 PRE-TRAINING COMPLETE!")
print("=" * 80)
print(f"📈 Best Val AUC: {best_auc:.4f}")
print(f"💾 Model: {model_path}")
print(f"📄 Metadata: {metadata_path}")
print(f"📊 Articles: {len(news_training):,}")
print(f"🎓 Developer: Prateek (IIT Patna MTech AI)")
print(f"🏢 Organization: Times Network")
print(f"📧 Contact: prat.cann.170701@gmail.com")
print("=" * 80)
