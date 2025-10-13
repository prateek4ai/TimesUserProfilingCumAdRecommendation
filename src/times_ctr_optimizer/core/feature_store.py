"""
Advanced Feature Engineering and Feature Store
Professional feature engineering with sequence modeling and monetization focus
"""

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class CTRFeatureStore:
    """
    Advanced feature store for CTR optimization
    Handles sequence features, embeddings, and temporal aggregates
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize feature store with configuration"""
        self.config = config or {
            'MAX_SEQ_LEN': 100,
            'EMBEDDING_DIM': 50,
            'N_BUCKETS': 10,
            'SAMPLE_NEGATIVES': True
        }
        
    def create_sequence_features(self, events_df: pl.DataFrame, 
                                max_seq_len: int = 100) -> pl.DataFrame:
        """
        Create user behavior sequences for attention-based modeling
        
        Args:
            events_df: Events DataFrame
            max_seq_len: Maximum sequence length
            
        Returns:
            User sequences DataFrame
        """
        print("🎯 Creating Sequence Features...")
        
        # Sort events by user and timestamp
        events_sorted = events_df.sort(['user_id', 'timestamp'])
        
        # Get first N events per user using groupby.head(N)
        events_topn = events_sorted.group_by('user_id').head(max_seq_len)
        
        # Aggregate to lists - Use pl.col directly (auto-aggregates to lists)
        user_sequences = events_topn.group_by('user_id').agg([
            pl.col('item_id'),  # Automatically aggregates to list
            pl.col('clicked'),  # Automatically aggregates to list
            pl.col('timestamp'),  # Automatically aggregates to list
            pl.col('dwell_time_ms'),  # Automatically aggregates to list
            pl.col('position'),  # Automatically aggregates to list
            pl.col('item_id').len().alias('sequence_length')
        ])
        
        # Rename columns for clarity
        user_sequences = user_sequences.rename({
            'item_id': 'item_sequence',
            'clicked': 'click_sequence',
            'timestamp': 'time_sequence',
            'dwell_time_ms': 'dwell_sequence',
            'position': 'position_sequence'
        })
        
        # Clip sequence length
        user_sequences = user_sequences.with_columns([
            pl.col('sequence_length').clip(upper_bound=max_seq_len).alias('sequence_length_clipped')
        ])
        
        print(f"✅ Created sequences for {len(user_sequences):,} users")
        return user_sequences
    
    def create_temporal_aggregates(self, events_df: pl.DataFrame, 
                                 items_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Create time-windowed aggregate features
        
        Args:
            events_df: Events DataFrame
            items_df: Items DataFrame
            
        Returns:
            Tuple of (user_features, item_features)
        """
        print("⏰ Creating Temporal Aggregates...")
        
        # Join events with items for sponsored/revenue features
        events_with_items = events_df.join(items_df, on='item_id', how='left')
        
        # User-level aggregates
        user_features = events_with_items.group_by('user_id').agg([
            # CTR features
            pl.col('clicked').mean().alias('user_ctr_overall'),
            pl.col('clicked').sum().alias('user_total_clicks'),
            pl.len().alias('user_total_impressions'),
            
            # Sponsored interaction features
            (pl.col('clicked') * pl.col('is_sponsored')).mean().alias('user_sponsored_ctr'),
            pl.col('is_sponsored').mean().alias('user_sponsored_exposure_rate'),
            
            # Revenue features
            (pl.col('clicked') * pl.col('price')).sum().alias('user_gmv'),
            (pl.col('clicked') * pl.col('payout')).sum().alias('user_ad_revenue'),
            
            # Category diversity
            pl.col('category_l1').n_unique().alias('user_category_diversity'),
            pl.col('category_l2').n_unique().alias('user_subcategory_diversity'),
            
            # Device/context patterns
            pl.col('device_type').mode().first().alias('user_primary_device'),
            pl.col('is_business_hours').mean().alias('user_business_hours_rate'),
            pl.col('is_weekend').mean().alias('user_weekend_rate'),
            
            # Position bias
            pl.col('position').mean().alias('user_avg_position_seen'),
            pl.when(pl.col('clicked').sum() > 0)
              .then((pl.col('clicked') * pl.col('position')).sum() / pl.col('clicked').sum())
              .otherwise(0)
              .alias('user_avg_click_position'),
        ])
        
        # Item-level aggregates
        item_features = events_with_items.group_by('item_id').agg([
            # Performance metrics
            pl.col('clicked').mean().alias('item_ctr'),
            pl.col('clicked').sum().alias('item_total_clicks'), 
            pl.len().alias('item_total_impressions'),
            
            # User engagement
            pl.col('dwell_time_ms').mean().alias('item_avg_dwell'),
            pl.col('user_id').n_unique().alias('item_unique_users'),
            
            # Position performance
            pl.col('position').mean().alias('item_avg_position'),
            pl.when(pl.col('position') <= 5).then(pl.col('clicked').mean()).alias('item_ctr_top5'),
            
            # Recency
            pl.col('timestamp').max().alias('item_last_seen'),
        ])
        
        print(f"✅ User features: {user_features.shape}")
        print(f"✅ Item features: {item_features.shape}")
        
        return user_features, item_features
    
    def create_content_embeddings(self, items_df: pl.DataFrame, 
                                embedding_dim: int = 50) -> pl.DataFrame:
        """
        Create simplified content embeddings avoiding heavy sentence transformers
        
        Args:
            items_df: Items DataFrame
            embedding_dim: Embedding dimension
            
        Returns:
            Item embeddings DataFrame
        """
        print("🔤 Creating Content Embeddings...")
        
        try:
            # Simplified approach: use TF-IDF style embeddings
            items_pd = items_df.to_pandas()
            
            # Combine text features
            combined_text = (items_pd['title'] + ' ' + 
                           items_pd['description'] + ' ' + 
                           items_pd['category_l1'] + ' ' + 
                           items_pd['category_l2'])
            
            # Create TF-IDF vectors and reduce dimensions
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(combined_text.fillna(''))
            
            # Reduce to specified dimensions  
            svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix)
            
            # Create embedding dataframe
            embedding_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
            embedding_df = pl.DataFrame({
                'item_id': items_pd['item_id'].tolist(),
                **{col: embeddings[:, i] for i, col in enumerate(embedding_cols)}
            })
            
            print(f"✅ Generated {embeddings.shape[1]}-dim TF-IDF embeddings for {len(embedding_df):,} items")
            return embedding_df
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            print("Creating dummy embeddings for compatibility...")
            
            # Fallback: create dummy embeddings
            n_items = len(items_df)
            np.random.seed(42)
            dummy_embeddings = np.random.randn(n_items, embedding_dim)
            
            embedding_cols = [f'emb_{i}' for i in range(embedding_dim)]
            return pl.DataFrame({
                'item_id': items_df['item_id'],
                **{col: dummy_embeddings[:, i] for i, col in enumerate(embedding_cols)}
            })
    
    def create_exposure_buckets(self, events_df: pl.DataFrame, 
                              n_buckets: int = 10) -> pl.DataFrame:
        """
        Create exposure quantile buckets for counterfactual debiasing
        
        Args:
            events_df: Events DataFrame
            n_buckets: Number of exposure buckets
            
        Returns:
            User exposure DataFrame with propensity weights
        """
        print("📊 Creating Exposure Buckets...")
        
        # Calculate user exposure levels
        user_exposure = events_df.group_by('user_id').agg([
            pl.len().alias('total_impressions'),
            pl.col('clicked').sum().alias('total_clicks'),
        ]).with_columns([
            (pl.col('total_clicks') / pl.col('total_impressions')).alias('user_ctr')
        ])
        
        # Sort by impressions and assign buckets based on rank
        user_exposure = user_exposure.with_row_index('row_num')
        total_users = len(user_exposure)
        bucket_size = total_users // n_buckets
        
        # Create bucket assignments
        user_exposure = user_exposure.with_columns([
            (pl.col('row_num') // bucket_size).clip(upper_bound=n_buckets-1).alias('bucket_id')
        ]).with_columns([
            pl.concat_str([pl.lit('bucket_'), pl.col('bucket_id').cast(pl.Utf8)])
              .alias('exposure_bucket')
        ])
        
        # Calculate bucket statistics for propensity weights
        bucket_stats = user_exposure.group_by('exposure_bucket').agg([
            pl.len().alias('bucket_size'),
            pl.col('user_ctr').mean().alias('bucket_avg_ctr')
        ])
        
        bucket_stats = bucket_stats.with_columns([
            (total_users / pl.col('bucket_size')).alias('propensity_weight')
        ])
        
        # Join back propensity weights
        user_exposure = user_exposure.join(
            bucket_stats.select(['exposure_bucket', 'propensity_weight']), 
            on='exposure_bucket', 
            how='left'
        )
        
        print(f"✅ Created {n_buckets} exposure buckets with propensity weights")
        return user_exposure.select(['user_id', 'exposure_bucket', 'propensity_weight'])
    
    def assemble_feature_store(self, events_df: pl.DataFrame, 
                              items_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Combine all features into final feature store
        
        Args:
            events_df: Events DataFrame
            items_df: Items DataFrame
            
        Returns:
            Tuple of (user_store, item_store)
        """
        print("🏗️ Assembling Feature Store...")
        
        # Generate all feature components
        user_sequences = self.create_sequence_features(events_df)
        user_features, item_features = self.create_temporal_aggregates(events_df, items_df)
        item_embeddings = self.create_content_embeddings(items_df)
        exposure_features = self.create_exposure_buckets(events_df)
        
        # Combine user-level features
        user_store = user_features.join(user_sequences, on='user_id', how='left')
        user_store = user_store.join(exposure_features, on='user_id', how='left')
        
        # Combine item-level features
        item_store = items_df.join(item_features, on='item_id', how='left')
        item_store = item_store.join(item_embeddings, on='item_id', how='left')
        
        # Add current timestamp for freshness tracking
        from datetime import datetime
        current_time = datetime.now()
        user_store = user_store.with_columns([pl.lit(current_time).alias('feature_timestamp')])
        item_store = item_store.with_columns([pl.lit(current_time).alias('feature_timestamp')])
        
        print(f"✅ User feature store: {user_store.shape}")
        print(f"✅ Item feature store: {item_store.shape}")
        
        return user_store, item_store
    
        def prepare_training_data(self, events_df: pl.DataFrame, 
                            user_store: pl.DataFrame, 
                            item_store: pl.DataFrame,
                            sample_negatives: bool = True) -> pl.DataFrame:
        """
        Prepare final training dataset with all features
        
        Args:
            events_df: Events DataFrame
            user_store: User feature store
            item_store: Item feature store
            sample_negatives: Whether to sample negative examples
            
        Returns:
            Training dataset DataFrame
        """
        print("📝 Preparing Training Data...")
        
        # Start with events as base
        training_data = events_df.select([
            'user_id', 'item_id', 'timestamp', 'clicked', 'session_id',
            'device_type', 'ad_unit_type', 'position', 'geo_country',
            'hour', 'day_of_week', 'month', 'is_business_hours', 'is_weekend',
            'daily_budget', 'current_spend', 'target_ctr', 'budget_utilization'
        ])
        
        # Join user features (excluding sequences for now to save memory)
        user_features_slim = user_store.select([
            'user_id', 'user_ctr_overall', 'user_sponsored_ctr', 'user_sponsored_exposure_rate',
            'user_gmv', 'user_category_diversity', 'user_primary_device', 'user_business_hours_rate',
            'user_avg_position_seen', 'exposure_bucket', 'propensity_weight'
        ])
        training_data = training_data.join(user_features_slim, on='user_id', how='left')
        
        # Join item features (excluding embeddings for now)
        item_features_slim = item_store.select([
            'item_id', 'price', 'margin_pct', 'is_sponsored', 'cpc_bid', 'quality_score',
            'category_l1', 'category_l2', 'payout', 'item_ctr', 'item_total_impressions',
            'item_avg_dwell', 'item_unique_users'
        ])
        training_data = training_data.join(item_features_slim, on='item_id', how='left')
        
        # Simple balanced sampling without complex logic
        if sample_negatives:
            positives = training_data.filter(pl.col('clicked') == 1)
            negatives = training_data.filter(pl.col('clicked') == 0)
            
            # Take equal numbers of positives and negatives (or all we have)
            n_pos = len(positives)
            n_neg = len(negatives)
            n_take = min(n_pos, n_neg, 50000)  # Limit for memory
            
            if n_take > 0:
                pos_sample = positives.head(n_take)
                neg_sample = negatives.head(n_take)
                training_data = pl.concat([pos_sample, neg_sample])
                print(f"✅ Sampled to {len(training_data):,} examples (CTR: {training_data['clicked'].mean():.3f})")
        
        # Handle missing values
        training_data = training_data.fill_null(0)
        
        print(f"✅ Final training data: {training_data.shape}")
        return training_data


# Factory function for easy import
def create_feature_store(config: Optional[dict] = None) -> CTRFeatureStore:
    """Factory function to create CTR feature store"""
    return CTRFeatureStore(config)
