import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_transforms

app = FastAPI(title="Emotion and Engagement Detection API")

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
    
    # Initialize model
    model = MultiHeadCNN()
    
    # Load pre-trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "fer2013_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load transforms
    transform = get_transforms(train=False)
    
    print("Model loaded successfully")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Emotion and Engagement Detection API is running"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Apply transforms
        if transform:
            image = transform(image).unsqueeze(0)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            emotion_logits = outputs['emotion']
            emotion_probs = torch.softmax(emotion_logits, dim=1)
            emotion_idx = torch.argmax(emotion_probs, dim=1).item()
            confidence = emotion_probs[0][emotion_idx].item()
            
            # Map emotion index to label
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion = emotion_labels[emotion_idx]
            
            # Get engagement level if available
            engagement = None
            if 'engagement' in outputs:
                engagement_logits = outputs['engagement']
                engagement_probs = torch.softmax(engagement_logits, dim=1)
                engagement_idx = torch.argmax(engagement_probs, dim=1).item()
                engagement_levels = ['Very Low', 'Low', 'High', 'Very High']
                engagement = engagement_levels[engagement_idx]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "engagement": engagement
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
