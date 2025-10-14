# 🚀 Deployment Guide

## Local Development

###Install
pip install -r requirements.txt
###Train
python newnotebook.py
###Run API
cd deployment && python api.py


## Docker

docker build -t times-ctr-api .
docker run -p 8000:8000 times-ctr-api

## Production Checklist

- [ ] Models trained and validated
- [ ] Environment variables configured  
- [ ] HTTPS enabled
- [ ] Monitoring set up
- [ ] Rate limiting configured

---

**Support:** prat.cann.170701@gmail.com
