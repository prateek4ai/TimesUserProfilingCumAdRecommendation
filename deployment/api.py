from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Times CTR API")

class Request(BaseModel):
    user_id: int
    item_id: int

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"healthy": True}

@app.post("/predict")
def predict(req: Request):
    return {"user_id": req.user_id, "predicted_ctr": 0.15}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
