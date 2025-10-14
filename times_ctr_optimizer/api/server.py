"""FastAPI server factory"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from ..predictor import CTRPredictor


def create_app(model_path: str = "outputs/best_wide_deep_model.pt") -> FastAPI:
    """
    Create FastAPI application.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Times Network CTR API",
        description="Enterprise CTR prediction service",
        version="1.0.0"
    )
    
    # Initialize predictor
    predictor = CTRPredictor(model_path)
    
    class PredictRequest(BaseModel):
        user_id: int
        item_id: int
        position: Optional[int] = 0
        hour: Optional[int] = 12
        
    class PredictResponse(BaseModel):
        user_id: int
        item_id: int
        predicted_ctr: float
        confidence: float = 0.95
    
    @app.get("/")
    def root():
        return {"status": "healthy", "version": "1.0.0"}
    
    @app.get("/health")
    def health():
        return {"healthy": True, "model_loaded": True}
    
    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        ctr = predictor.predict(
            user_id=request.user_id,
            item_id=request.item_id,
            position=request.position,
            hour=request.hour
        )
        return PredictResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            predicted_ctr=ctr
        )
    
    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
