import re

# Read the file and find the problematic method
with open('times_ctr_optimizer/core/feature_store.py', 'r') as f:
    content = f.read()

# Replace the entire prepare_training_data method with a working version
method_replacement = '''    def prepare_training_data(self, events_df: pl.DataFrame, 
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
        return training_data'''

# Use regex to replace the method
pattern = r'def prepare_training_data\(self,.*?(?=\n    def|\n\n\n# Factory|\nclass|\Z)'
new_content = re.sub(pattern, method_replacement, content, flags=re.DOTALL)

# Write the fixed version
with open('times_ctr_optimizer/core/feature_store.py', 'w') as f:
    f.write(new_content)

print("✅ Fixed feature_store.py with clean sampling logic")
