# API Documentation

## Endpoints
- GET /health
- POST /predict

## Example
import requests
response = requests.post("http://localhost:8000/predict", json={"user_id": 12345, "item_id": 67890})
text
