# %% [code] {"execution":{"iopub.status.busy":"2025-09-17T08:39:49.398593Z","iopub.execute_input":"2025-09-17T08:39:49.398780Z","iopub.status.idle":"2025-09-17T08:39:51.092172Z","shell.execute_reply.started":"2025-09-17T08:39:49.398763Z","shell.execute_reply":"2025-09-17T08:39:51.091539Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (outputs/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2025-09-17T08:39:51.093026Z","iopub.execute_input":"2025-09-17T08:39:51.093692Z","iopub.status.idle":"2025-09-17T08:39:54.585303Z","shell.execute_reply.started":"2025-09-17T08:39:51.093666Z","shell.execute_reply":"2025-09-17T08:39:54.584586Z"}}
# =============================================================================
# CELL #1: RAW DATA SOURCES & INGESTION PIPELINE
# Optimized for Kaggle T4x2 Environment (13GB RAM, 20GB Storage)
# =============================================================================

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_CONFIG = {
    'MAX_USERS': 100_000,  # Cap for memory efficiency
    'MAX_ITEMS': 50_000,   # Cap for item catalog
    'LOOKBACK_DAYS': 30,   # Historical window
    'SAMPLE_RATE': 0.1,    # Sample rate for large datasets
}

print("🚀 CTR Optimization Pipeline - Raw Data Sources")
print("=" * 60)

# =============================================================================
# 1.1 USER EVENT STREAMS
# =============================================================================

def generate_user_events(n_users=10000, n_items=5000, n_events=500000):
    """
    Simulate user interaction events with monetized items
    """
    print("📊 Generating User Event Stream...")
    
    np.random.seed(42)
    
    # Generate base event data
    events_data = {
        'user_id': np.random.randint(1, n_users + 1, n_events),
        'item_id': np.random.randint(1, n_items + 1, n_events),
        'timestamp': pd.date_range('2025-08-01', periods=n_events, freq='1min'),
        'session_id': np.random.randint(1, n_events // 10, n_events),
        'event_type': np.random.choice(['impression', 'click', 'add_to_cart', 'purchase'], 
                                     n_events, p=[0.7, 0.2, 0.07, 0.03]),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], 
                                      n_events, p=[0.6, 0.3, 0.1]),
        'ad_unit_type': np.random.choice(['banner', 'native', 'video', 'search'], 
                                       n_events, p=[0.4, 0.3, 0.2, 0.1]),
        'creative_id': np.random.randint(1, 1000, n_events),
        'position': np.random.randint(1, 21, n_events),  # Ad position 1-20
        'geo_country': np.random.choice(['US', 'UK', 'DE', 'FR', 'IN'], 
                                      n_events, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'dwell_time_ms': np.random.exponential(2000, n_events).astype(int),
    }
    
    events_df = pl.DataFrame(events_data)
    
    # Add CTR label (binary click indicator) - FIXED datetime API
    events_df = events_df.with_columns([
        (pl.col('event_type') == 'click').cast(pl.Int8).alias('clicked'),
        pl.col('timestamp').dt.hour().alias('hour'),
        pl.col('timestamp').dt.weekday().alias('day_of_week'),  # FIXED: Use weekday() instead of day_of_week()
    ])
    
    print(f"✅ Generated {len(events_df):,} events for {n_users:,} users")
    return events_df

# =============================================================================
# 1.2 ITEM METADATA & MONETIZATION TAGS
# =============================================================================

def generate_item_metadata(n_items=5000):
    """
    Generate item catalog with monetization metadata
    """
    print("🏷️  Generating Item Metadata...")
    
    np.random.seed(42)
    
    # Category hierarchy
    categories_l1 = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books']
    categories_l2 = {
        'Electronics': ['Phones', 'Laptops', 'Audio', 'Gaming'],
        'Fashion': ['Clothing', 'Shoes', 'Accessories', 'Jewelry'],
# CELL #9: PRE-TRAINING ON PUBLIC NEWS DATASET (UCI News Aggregator)
# This demonstrates how the system can be pre-trained on a general news
# dataset to create a powerful baseline model before fine-tuning on specific data.
# Developed by: Prateek (MTech AI, IIT Patna | Times Network Intern)
# Email: prat.cann.170701@gmail.com
# =============================================================================

print("\n" + "=" * 80)
print("🚀 PRE-TRAINING ON PUBLIC NEWS DATASET")
print("🎓 IIT Patna MTech AI Project | 🏢 Times Network Application")
print("=" * 80)

import requests
import zipfile
import os
from pathlib import Path

# --- 1. Download and Prepare the Dataset ---
def download_and_prepare_news_data():
    """
    Downloads and prepares the UCI News Aggregator dataset.
    Adapts it to match the CTR optimization pipeline requirements.
    """
    print("\n📥 Downloading UCI News Aggregator dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip"
    
    # Use outputs directory for Codespaces
    data_dir = Path("outputs")
    data_dir.mkdir(exist_ok=True)
    
    zip_path = data_dir / "NewsAggregatorDataset.zip"
    data_path = data_dir / "newsCorpora.csv"
    
    if not data_path.exists():
        # Download the file
        print("   Downloading from UCI ML Repository...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Unzip the file
        print("   Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract("newsCorpora.csv", str(data_dir))
        
        print("✅ Dataset downloaded and extracted.")
    else:
        print("✅ Dataset already available.")

    # --- 2. Load and Adapt the Data Schema ---
    print("\n🔧 Adapting news data schema for CTR pipeline...")
    
    # The dataset is tab-separated with no header
    # Columns: ID, TITLE, URL, PUBLISHER, CATEGORY, STORY, HOSTNAME, TIMESTAMP
    news_df = pl.read_csv(
        str(data_path),
        separator='\t',
        has_header=False,
        new_columns=["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
    )
    
    print(f"   Loaded {len(news_df):,} news articles")

    # Map categories to meaningful names
    category_map = {
        'b': 'Business', 
        't': 'SciTech', 
        'e': 'Entertainment', 
        'm': 'Health'
    }
    
    # Create adapted dataframe with required columns
    news_df = news_df.with_columns([
        pl.col("TITLE").alias("title"),
        pl.col("CATEGORY").replace(category_map).alias("category_l1"),
        pl.col("PUBLISHER").alias("publisher"),
        # Generate sequential item IDs
        pl.int_range(0, pl.len()).alias("item_id"),
        # Simulate user IDs (distribute across 10k simulated users)
        (pl.int_range(0, pl.len()) % 10000).alias("user_id")
    ])

    # --- 3. Simulate Required Features for Training ---
    print("⚙️  Simulating CTR features (clicks, sponsored status, temporal features)...")
    
    # Simulate realistic click patterns
    # Higher CTR for certain categories (e.g., Entertainment)
    np.random.seed(42)
    base_ctr = 0.08  # 8% base CTR
    
    # Category-specific CTR adjustments
    category_boost = {
        'Entertainment': 0.05,
        'SciTech': 0.03,
        'Business': 0.02,
        'Health': 0.04
    }
    
    clicks = []
    for cat in news_df['category_l1']:
        ctr = base_ctr + category_boost.get(cat, 0.0)
        clicks.append(np.random.binomial(1, ctr))
    
    # Simulate sponsored items (15% are sponsored)
    all_item_ids = news_df['item_id'].unique().to_list()
    n_sponsored = int(len(all_item_ids) * 0.15)
    sponsored_items = set(np.random.choice(all_item_ids, n_sponsored, replace=False))
    
    # Add all required features
    news_df = news_df.with_columns([
        pl.Series("clicked", clicks),
        pl.col("item_id").is_in(sponsored_items).alias("is_sponsored"),
        # Position features
        (pl.int_range(0, pl.len()) % 10).alias("position"),
        # Temporal features
        (pl.int_range(0, pl.len()) % 24).alias("hour"),
        (pl.int_range(0, pl.len()) % 7).alias("day_of_week"),
        pl.lit(10).alias("month"),  # October
        # Pricing (sponsored items cost more)
        pl.when(pl.col("item_id").is_in(sponsored_items))
          .then(pl.lit(10.0))
          .otherwise(pl.lit(0.0))
          .alias("price"),
        # User features (will be computed)
        pl.lit(0.0).alias("user_ctr_overall"),
        pl.lit(0.0).alias("user_sponsored_ctr"),
        # Item features (will be computed)
        pl.lit(0.0).alias("item_ctr"),
        # Categorical features
        pl.lit("mobile").alias("device_type"),
        pl.lit("standard").alias("ad_unit_type"),
        pl.lit("IN").alias("geo_country"),
        pl.col("category_l1").alias("category_l2"),
        pl.lit("mobile").alias("user_primary_device"),
        pl.lit("high").alias("exposure_bucket")
    ])

    # Compute actual user and item CTRs from data
    user_ctrs = news_df.group_by("user_id").agg([
        pl.col("clicked").mean().alias("user_ctr_overall")
    ])
    
    item_ctrs = news_df.group_by("item_id").agg([
        pl.col("clicked").mean().alias("item_ctr")
    ])
    
    # Join back to main dataframe
    news_df = news_df.drop(["user_ctr_overall", "item_ctr"])
    news_df = news_df.join(user_ctrs, on="user_id", how="left")
    news_df = news_df.join(item_ctrs, on="item_id", how="left")
    
    # Fill any missing values
    news_df = news_df.with_columns([
        pl.col("user_ctr_overall").fill_null(0.08),
        pl.col("item_ctr").fill_null(0.08)
    ])

    print(f"✅ Prepared {len(news_df):,} records for pre-training.")
    print(f"   📊 Overall CTR: {news_df['clicked'].mean():.3f}")
    print(f"   💰 Sponsored ratio: {news_df['is_sponsored'].mean():.3f}")
    
    return news_df

# Execute the data preparation
news_training_data = download_and_prepare_news_data()

# --- 4. Train a Model on the News Data ---
print("\n🚀 Training CTR model on public news dataset...")
print("   This creates a 'news expert' baseline model")

# Use the processor from earlier cells
news_processor = CTRDataProcessor()
print("   Processing features...")
features, labels, weights = news_processor.fit_transform(news_training_data)

print(f"   Feature dimensions: {features.shape}")
print(f"   Positive samples: {labels.sum():,} ({labels.mean():.3f})")

# Create datasets and loaders
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
    features, labels, weights, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = CTRDataset(X_train, y_train, w_train)
val_dataset = CTRDataset(X_val, y_val, w_val)

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

print(f"   Training samples: {len(train_dataset):,}")
print(f"   Validation samples: {len(val_dataset):,}")

# Initialize and train the model
print("\n🧠 Initializing Wide & Deep model for news domain...")
news_model = WideDeepModel(
    categorical_dims=news_processor.categorical_dims,
    num_numerical=news_processor.num_numerical
)

print(f"   Model parameters: {sum(p.numel() for p in news_model.parameters()):,}")

# Train the model
print("\n🏋️ Training on news dataset (5 epochs)...")
best_metric = train_model(news_model, train_loader, val_loader, epochs=5)

# --- 5. Save the Pre-trained Model ---
model_path = Path('outputs') / 'news_pretrained_model.pth'
torch.save(news_model.state_dict(), model_path)

print(f"\n{'='*80}")
print(f"✅ PRE-TRAINING COMPLETE!")
print(f"{'='*80}")
print(f"📈 Best Validation AUC on News Data: {best_metric:.4f}")
print(f"💾 Pre-trained 'news expert' model saved to: {model_path}")
print(f"\n🎓 Model Details:")
print(f"   - Trained on {len(news_training_data):,} news articles")
print(f"   - UCI News Aggregator dataset")
print(f"   - Categories: Business, SciTech, Entertainment, Health")
print(f"   - Ready for transfer learning on Times Network data")
print(f"\n👨‍🎓 Developed by: Prateek (IIT Patna MTech AI)")
print(f"🏢 Industry Application: Times Network Internship")
print(f"📧 Contact: prat.cann.170701@gmail.com")
print(f"{'='*80}\n")

# Save training metadata
import json
metadata = {
    "model_type": "WideDeepModel",
    "dataset": "UCI News Aggregator",
    "training_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "best_auc": float(best_metric),
    "categories": ["Business", "SciTech", "Entertainment", "Health"],
    "overall_ctr": float(news_training_data['clicked'].mean()),
    "sponsored_ratio": float(news_training_data['is_sponsored'].mean()),
    "developer": "Prateek (IIT Patna MTech AI | Times Network)",
    "email": "prat.cann.170701@gmail.com",
    "date_trained": "2025-10-14"
}

metadata_path = Path('outputs') / 'news_model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"📄 Training metadata saved to: {metadata_path}")

