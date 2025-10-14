from times_ctr_optimizer import CTRPredictor

# Initialize
predictor = CTRPredictor(model_path="outputs/best_wide_deep_model.pt")

# Predict
ctr = predictor.predict(user_id=12345, item_id=67890)
print(f"Predicted CTR: {ctr:.2%}")
