"""
FastAPI server for retinal disease classification model inference
"""
import os
import sys
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
from typing import Dict, List, Optional
import logging
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Import model architecture
from models.vignn import ViGNN, create_vignn_model, create_knowledge_graph

# Pydantic models for request/response validation
class PredictionResult(BaseModel):
    disease: str
    probability: float
    confidence: str

class PredictionResponse(BaseModel):
    success: bool
    predictions: List[PredictionResult]
    total_diseases_detected: int
    all_probabilities: Dict[str, float]
    model_loaded: bool
    processing_time_ms: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    diseases_count: int

class DiseasesResponse(BaseModel):
    total_diseases: int
    diseases: List[str]

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Retinal Disease Classification API",
    description="Multi-label retinal disease classification using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
MODEL = None
DEVICE = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_vignn_rank1.pth")
METADATA_PATH = "models/model_vignn_rank1_metadata.json"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Disease labels (will be set from knowledge graph)
DISEASE_LABELS = []


@app.on_event("startup")
async def load_model():
    """Load model on startup with optimizations"""
    global MODEL, DEVICE, DISEASE_LABELS

    try:
        # Device detection
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {DEVICE}")

        # Create knowledge graph first
        logger.info("Creating ClinicalKnowledgeGraph...")
        kg = create_knowledge_graph()
        DISEASE_LABELS = kg.disease_names
        logger.info(f"✓ Knowledge graph created with {len(DISEASE_LABELS)} diseases")

        if os.path.exists(MODEL_PATH):
            logger.info(f"Found model file at {MODEL_PATH}")

            # Use the create_vignn_model function which handles checkpoint loading
            MODEL = create_vignn_model(
                num_classes=len(DISEASE_LABELS),
                clinical_knowledge_graph=kg,
                checkpoint_path=MODEL_PATH
            )

            # Move to device and set to eval mode
            MODEL.to(DEVICE)
            MODEL.eval()

            # Memory optimizations for GPU
            if DEVICE.type == "cuda":
                try:
                    # Enable attention slicing to reduce memory usage
                    if hasattr(MODEL, 'enable_attention_slicing'):
                        MODEL.enable_attention_slicing()
                        logger.info("✓ Attention slicing enabled")

                    # Try to enable xformers for memory efficiency
                    try:
                        from xformers.ops import memory_efficient_attention
                        logger.info("✓ xFormers available for memory efficient attention")
                    except ImportError:
                        logger.info("xFormers not available, using standard attention")

                except Exception as opt_error:
                    logger.warning(f"Could not apply memory optimizations: {opt_error}")

            logger.info("✓ ViGNN model loaded successfully")

            # Load and log metadata if available
            if os.path.exists(METADATA_PATH):
                try:
                    with open(METADATA_PATH, 'r') as f:
                        metadata = json.load(f)
                    logger.info("✓ Metadata loaded from {METADATA_PATH}")
                    if 'best_f1' in metadata:
                        logger.info(f"  Best F1 Score: {metadata['best_f1']:.4f}")
                    if 'best_auc' in metadata:
                        logger.info(f"  Best AUC Score: {metadata['best_auc']:.4f}")
                    if 'epoch' in metadata:
                        logger.info(f"  Trained for {metadata['epoch']} epochs")
                except Exception as meta_error:
                    logger.warning(f"Could not load metadata: {meta_error}")

        else:
            logger.warning(f"Model not found at {MODEL_PATH}")
            logger.warning("API will run in DEMO MODE with random predictions")
            MODEL = None

    except Exception as e:
        logger.error(f"Unexpected error in startup: {e}")
        logger.exception("Full traceback:")
        MODEL = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global MODEL
    if MODEL is not None:
        logger.info("Cleaning up model resources...")
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MODEL = None
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Retinal Disease Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": MODEL is not None,
        "diseases_count": len(DISEASE_LABELS)
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration"""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE) if DEVICE else "not initialized",
        diseases_count=len(DISEASE_LABELS)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Predict retinal diseases from uploaded image

    Args:
        file: Uploaded image file (JPG, PNG)

    Returns:
        JSON with predictions and probabilities
    """
    import time
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Validate file size (max 10MB)
        file_size = len(await file.read())
        await file.seek(0)  # Reset file pointer
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")

        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Validate image dimensions
        if image.width < 32 or image.height < 32:
            raise HTTPException(status_code=400, detail="Image too small (min 32x32)")

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
                predictions.append(PredictionResult(
                    disease=DISEASE_LABELS[idx],
                    probability=float(prob),
                    confidence="high" if prob > 0.8 else "medium"
                ))

        # Sort by probability
        predictions.sort(key=lambda x: x.probability, reverse=True)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return PredictionResponse(
            success=True,
            predictions=predictions,
            total_diseases_detected=len(predictions),
            all_probabilities={
                DISEASE_LABELS[i]: float(probabilities[i])
                for i in range(len(DISEASE_LABELS))
            },
            model_loaded=MODEL is not None,
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        processing_time = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)} (processed in {processing_time:.2f}ms)"
        )


@app.get("/diseases", response_model=DiseasesResponse)
async def list_diseases():
    """List all detectable diseases"""
    return DiseasesResponse(
        total_diseases=len(DISEASE_LABELS),
        diseases=DISEASE_LABELS
    )


if __name__ == "__main__":
    # Validate configuration
    if not DISEASE_LABELS:
        logger.warning("No disease labels loaded - API may not function correctly")

    # Get port from environment or use default
    port = int(os.getenv("PORT", str(API_PORT)))

    logger.info(f"Starting server on {API_HOST}:{port}")
    logger.info(f"Model loaded: {MODEL is not None}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Diseases: {len(DISEASE_LABELS)}")

    uvicorn.run(
        app,
        host=API_HOST,
        port=port,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )
