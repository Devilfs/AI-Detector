"""Main FastAPI application for AI Content Detection System."""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import time
from PIL import Image
import io
import uvicorn

from src.text_detection import TextEnsemble
from src.image_detection import ImageEnsemble
from src.utils import setup_logger, get_api_config

# Setup logging
logger = setup_logger(__name__)

# Load configuration
config = get_api_config()

# Initialize FastAPI app
app = FastAPI(
    title="AI Content Detection API",
    description="A hybrid AI-generated text and image detector for high-accuracy, low-cost, and production-scale deployment.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
text_ensemble = None
image_ensemble = None


# Pydantic models
class TextInput(BaseModel):
    text: str
    detailed: Optional[bool] = False


class TextBatchInput(BaseModel):
    texts: List[str]
    detailed: Optional[bool] = False


class PredictionResponse(BaseModel):
    is_ai: bool
    confidence: float
    method: str
    processing_time_ms: Optional[float] = None


class DetailedPredictionResponse(PredictionResponse):
    individual_predictions: Optional[dict] = None
    detailed_features: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: dict
    device_info: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    status_code: int


# Dependency to get text ensemble
async def get_text_ensemble():
    global text_ensemble
    if text_ensemble is None:
        try:
            text_ensemble = TextEnsemble()
            logger.info("Text ensemble loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text ensemble: {e}")
            raise HTTPException(status_code=500, detail="Text detection models not available")
    return text_ensemble


# Dependency to get image ensemble
async def get_image_ensemble():
    global image_ensemble
    if image_ensemble is None:
        try:
            image_ensemble = ImageEnsemble()
            logger.info("Image ensemble loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load image ensemble: {e}")
            raise HTTPException(status_code=500, detail="Image detection models not available")
    return image_ensemble


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting AI Content Detection API...")
    
    # Pre-load models
    try:
        await get_text_ensemble()
        await get_image_ensemble()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.warning(f"Some models failed to load: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Content Detection API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "text_detection": "/detect/text",
            "image_detection": "/detect/image",
            "batch_text": "/detect/text/batch",
            "batch_image": "/detect/image/batch"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from src.utils import get_device_info
    
    models_status = {
        "text_ensemble": text_ensemble is not None,
        "image_ensemble": image_ensemble is not None
    }
    
    device_info = None
    try:
        device_info = get_device_info()
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
    
    return HealthResponse(
        status="healthy" if any(models_status.values()) else "unhealthy",
        version="1.0.0",
        models_loaded=models_status,
        device_info=device_info
    )


@app.post("/detect/text", response_model=DetailedPredictionResponse)
async def detect_text(
    input_data: TextInput,
    ensemble: TextEnsemble = Depends(get_text_ensemble)
):
    """Detect if text is AI-generated."""
    start_time = time.time()
    
    try:
        # Validate input
        if not input_data.text or len(input_data.text.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Text must be at least 10 characters long"
            )
        
        # Get prediction
        if input_data.detailed:
            result = ensemble.analyze_detailed(input_data.text)
            response_data = result['basic_prediction']
            response_data['detailed_features'] = result.get('detailed_features', {})
        else:
            response_data = ensemble.predict(input_data.text)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        response_data['processing_time_ms'] = processing_time
        
        return DetailedPredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/detect/text/batch", response_model=List[DetailedPredictionResponse])
async def detect_text_batch(
    input_data: TextBatchInput,
    ensemble: TextEnsemble = Depends(get_text_ensemble)
):
    """Detect if multiple texts are AI-generated."""
    start_time = time.time()
    
    try:
        # Validate input
        if not input_data.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(input_data.texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
        
        # Get predictions
        results = ensemble.predict_batch(input_data.texts)
        
        # Add processing time to each result
        processing_time = (time.time() - start_time) * 1000
        for result in results:
            result['processing_time_ms'] = processing_time / len(input_data.texts)
        
        return [DetailedPredictionResponse(**result) for result in results]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch text detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/detect/image", response_model=DetailedPredictionResponse)
async def detect_image(
    file: UploadFile = File(...),
    detailed: bool = False,
    ensemble: ImageEnsemble = Depends(get_image_ensemble)
):
    """Detect if image is AI-generated."""
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_types = config.get('allowed_image_types', ['jpg', 'jpeg', 'png', 'webp'])
        file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Check file size
        max_size = config.get('max_file_size', 10485760)  # 10MB default
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size / 1024 / 1024:.1f}MB"
            )
        
        # Load and process image
        image = Image.open(io.BytesIO(content)).convert('RGB')
        
        # Get prediction
        if detailed:
            result = ensemble.analyze_detailed(image)
            response_data = result['basic_prediction']
            response_data['detailed_features'] = result.get('detailed_features', {})
        else:
            response_data = ensemble.predict(image)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        response_data['processing_time_ms'] = processing_time
        
        return DetailedPredictionResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/detect/image/batch", response_model=List[DetailedPredictionResponse])
async def detect_image_batch(
    files: List[UploadFile] = File(...),
    detailed: bool = False,
    ensemble: ImageEnsemble = Depends(get_image_ensemble)
):
    """Detect if multiple images are AI-generated."""
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(files) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 50)")
        
        allowed_types = config.get('allowed_image_types', ['jpg', 'jpeg', 'png', 'webp'])
        max_size = config.get('max_file_size', 10485760)
        
        images = []
        
        # Process all files
        for i, file in enumerate(files):
            # Validate file
            file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
            if file_extension not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i+1}: Unsupported file type. Allowed: {', '.join(allowed_types)}"
                )
            
            content = await file.read()
            if len(content) > max_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {i+1}: File too large. Maximum size: {max_size / 1024 / 1024:.1f}MB"
                )
            
            # Load image
            image = Image.open(io.BytesIO(content)).convert('RGB')
            images.append(image)
        
        # Get predictions
        results = ensemble.predict_batch(images)
        
        # Add processing time to each result
        processing_time = (time.time() - start_time) * 1000
        for result in results:
            result['processing_time_ms'] = processing_time / len(files)
        
        return [DetailedPredictionResponse(**result) for result in results]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch image detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/models/info")
async def get_models_info(
    text_ensemble: TextEnsemble = Depends(get_text_ensemble),
    image_ensemble: ImageEnsemble = Depends(get_image_ensemble)
):
    """Get information about loaded models."""
    try:
        info = {
            "text_models": text_ensemble.get_model_info() if text_ensemble else None,
            "image_models": image_ensemble.get_model_info() if image_ensemble else None
        }
        return info
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred",
            status_code=500
        ).dict()
    )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000),
        reload=True,
        log_level="info"
    )