import os

os.makedirs('docs', exist_ok=True)

readme = """# 🚀 Times Network CTR Optimization System

**Enterprise-grade CTR prediction for Times Network**

Developer: Prateek | IIT Patna MTech AI | prat.cann.170701@gmail.com

## Performance
- Model AUC: 87.46%
- API Latency: <50ms
- Throughput: 1000+ req/s

## Quick Start
pip install -r requirements.txt
python newnotebook.py
cd deployment && python api.py

## API Usage
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"user_id": 12345, "item_id": 67890}'

## Author
Prateek | IIT Patna MTech AI | Times Network
"""

api = """# API Documentation

## Endpoints

### GET /health
Returns: {"healthy": true}

### POST /predict
Request: {"user_id": 12345, "item_id": 67890}
Response: {"user_id": 12345, "item_id": 67890, "predicted_ctr": 0.15}

## Example
import requests
requests.post("http://localhost:8000/predict", json={"user_id": 12345, "item_id": 67890})
"""

deploy = """# Deployment Guide

## Local
pip install -r requirements.txt
cd deployment && python api.py

## Docker
docker build -t times-ctr-api .
docker run -p 8000:8000 times-ctr-api
"""

with open('README.md', 'w') as f: f.write(readme)
with open('docs/API.md', 'w') as f: f.write(api)
with open('docs/DEPLOYMENT.md', 'w') as f: f.write(deploy)
    
print("✅ Done! Created README.md, docs/API.md, docs/DEPLOYMENT.md")
