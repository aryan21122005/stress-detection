import os
import sys
import torch
import logging
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from src.models.multihead_cnn import MultiHeadCNN
    from src.utils.transforms import get_transforms
except ImportError as e:
    logger.error(f"Failed to import project modules: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

app = FastAPI(
    title="Emotion and Engagement Detection API",
    description="API for detecting emotions and engagement levels from facial images",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model and transforms
model = None
transform = None

def load_model():
    global model, transform
    
    try:
        logger.info("Initializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model
        model = MultiHeadCNN()
        
        # Load pre-trained weights
        model_path = os.path.join(project_root, "checkpoints", "fer2013_best.pth")
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Load with map_location to handle device compatibility
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Load transforms
        transform = get_transforms(train=False)
        
        logger.info("Model and transforms loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Don't raise here to allow the API to start for health checks

# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    }

# Health check endpoint with model status
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    return {"status": "healthy"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
        
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read and preprocess image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file provided")
            
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply transforms
        try:
            if transform:
                image_tensor = transform(image).unsqueeze(0)
            else:
                raise HTTPException(status_code=500, detail="Image transforms not initialized")
        except Exception as e:
            logger.error(f"Error transforming image: {str(e)}")
            raise HTTPException(status_code=500, detail="Error processing image")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(image_tensor)
                
                # Process emotion output
                if 'emotion' not in outputs:
                    raise HTTPException(status_code=500, detail="Model output format unexpected - missing 'emotion' key")
                    
                emotion_logits = outputs['emotion']
                emotion_probs = torch.softmax(emotion_logits, dim=1)
                emotion_idx = torch.argmax(emotion_probs, dim=1).item()
                confidence = emotion_probs[0][emotion_idx].item()
                
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion = emotion_labels[emotion_idx] if emotion_idx < len(emotion_labels) else "Unknown"
                
                # Process engagement if available
                engagement = None
                if 'engagement' in outputs:
                    engagement_logits = outputs['engagement']
                    engagement_probs = torch.softmax(engagement_logits, dim=1)
                    engagement_idx = torch.argmax(engagement_probs, dim=1).item()
                    engagement_levels = ['Very Low', 'Low', 'High', 'Very High']
                    engagement = engagement_levels[engagement_idx] if engagement_idx < len(engagement_levels) else "Unknown"
            
            return {
                "status": "success",
                "predictions": {
                    "emotion": {
                        "label": emotion,
                        "confidence": round(confidence, 4)
                    },
                    "engagement": {
                        "level": engagement,
                        "confidence": round(float(torch.max(engagement_probs).item()), 4) if engagement else None
                    } if engagement else None
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
