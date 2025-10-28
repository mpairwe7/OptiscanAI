"""
FastAPI server for retinal disease classification model inference
"""
import os
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Retinal Disease Classification API",
    description="Multi-label retinal disease classification using deep learning",
    version="1.0.0"
)

# Global model variable
MODEL = None
DEVICE = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/exports/best_model.pth")

# Disease labels (45 classes)
DISEASE_LABELS = [
    "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
    "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
    "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "CWS", "CB", "ODPM",
    "PRH", "MNF", "HR", "CRAO", "TD", "CME", "PTCR", "CF", "VH", "MCA",
    "VS", "BRAO", "PLQ", "HPED", "CL"
]


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global MODEL, DEVICE
    
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {DEVICE}")
        
        if os.path.exists(MODEL_PATH):
            # Load model checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            MODEL = checkpoint.get('model', checkpoint)  # Handle different checkpoint formats
            MODEL.to(DEVICE)
            MODEL.eval()
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. API will run in demo mode.")
            MODEL = None
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        MODEL = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Retinal Disease Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "not initialized"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict retinal diseases from uploaded image
    
    Args:
        file: Uploaded image file (JPG, PNG)
        
    Returns:
        JSON with predictions and probabilities
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to model input size
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        
        # Make prediction
        if MODEL is not None:
            with torch.no_grad():
                output = MODEL(image_tensor)
                probabilities = torch.sigmoid(output).cpu().numpy()[0]
        else:
            # Demo mode - return random predictions
            logger.warning("Running in demo mode - returning random predictions")
            probabilities = np.random.rand(len(DISEASE_LABELS))
        
        # Get top predictions (threshold > 0.5)
        predictions = []
        for idx, prob in enumerate(probabilities):
            if prob > 0.5:
                predictions.append({
                    "disease": DISEASE_LABELS[idx],
                    "probability": float(prob),
                    "confidence": "high" if prob > 0.8 else "medium"
                })
        
        # Sort by probability
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return JSONResponse({
            "success": True,
            "predictions": predictions,
            "total_diseases_detected": len(predictions),
            "all_probabilities": {
                DISEASE_LABELS[i]: float(probabilities[i]) 
                for i in range(len(DISEASE_LABELS))
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/diseases")
async def list_diseases():
    """List all detectable diseases"""
    return {
        "total_diseases": len(DISEASE_LABELS),
        "diseases": DISEASE_LABELS
    }


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
