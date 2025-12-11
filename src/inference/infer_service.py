import io
import time
from typing import Dict
import os
import json
import urllib.request
import urllib.error
from pymongo import MongoClient

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_inference_transform

EMOTION_LABELS = [
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
]

app = FastAPI(title="Local Inference Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_tf = get_inference_transform(224)
_mongo_client = None
_mongo_db = None
_mongo_coll = None


def _load_model(weights_path: str | None = None):
    global _model
    if _model is not None:
        return _model
    model = MultiHeadCNN()
    if weights_path:
        try:
            ckpt = torch.load(weights_path, map_location=_device)
            sd = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass
    model.eval().to(_device)
    _model = model
    return _model


@app.on_event("startup")
def _startup():
    # Lazy load on first request as well, but try early to warm up
    _load_model()
    # Initialize Mongo if env vars are present
    global _mongo_client, _mongo_db, _mongo_coll
    uri = os.getenv("MONGO_URI")
    dbname = os.getenv("MONGO_DB")
    collname = os.getenv("MONGO_COLLECTION", "analytics_data")
    if uri and dbname:
        try:
            _mongo_client = MongoClient(uri)
            _mongo_db = _mongo_client[dbname]
            _mongo_coll = _mongo_db[collname]
        except Exception:
            _mongo_client = None
            _mongo_db = None
            _mongo_coll = None


def _softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)


def _adjust_stress(emo_label: str, emo_conf: float, stress_probs: torch.Tensor) -> torch.Tensor:
    # Follow webcam_app.py logic: adjust only if high confidence
    if emo_conf >= 0.6:
        if emo_label in ["happy", "surprise"]:
            adj = torch.tensor([2.0, 0.5, 0.2, 0.1], device=stress_probs.device)
            return stress_probs * adj
        elif emo_label in ["sad", "angry", "disgust", "fear"]:
            adj = torch.tensor([0.6, 0.9, 1.2, 1.4], device=stress_probs.device)
            return stress_probs * adj
    return stress_probs


@app.post("/infer")
async def infer(file: UploadFile = File(...), weights: str | None = None) -> Dict:
    t0 = time.time()
    model = _load_model(weights)
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        emo_logits, eng_logits, stress_logits = model(x)
        emo_probs = _softmax(emo_logits)[0]
        eng_probs = _softmax(eng_logits)[0]
        stress_probs = _softmax(stress_logits)[0]

    emo_idx = int(torch.argmax(emo_probs).item())
    emo_label = EMOTION_LABELS[emo_idx]
    emo_conf = float(emo_probs[emo_idx].item())

    stress_probs = _adjust_stress(emo_label, emo_conf, stress_probs)
    stress_probs = stress_probs / (stress_probs.sum() + 1e-8)

    resp = {
        "emotion": {"label": emo_label, "confidence": emo_conf, "probs": emo_probs.cpu().tolist()},
        "engagement": {"probs": eng_probs.cpu().tolist()},
        "stress": {"probs": stress_probs.cpu().tolist()},
        "latency_ms": int((time.time() - t0) * 1000),
    }
    return resp


@app.post("/metrics")
async def save_metrics(payload: Dict) -> Dict:
    """
    Secure relay endpoint: accepts a JSON payload with fields
    { session_id, timestamp?, emotion, stress, attention }
    and writes a row to Supabase public.analytics_data using server-side creds.
    Requires environment variables:
      SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
    """
    global _mongo_coll
    if _mongo_coll is None:
        # Late init if not already done
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DB")
        collname = os.getenv("MONGO_COLLECTION", "analytics_data")
        if uri and dbname:
            try:
                _client = MongoClient(uri)
                _db = _client[dbname]
                _mongo_coll = _db[collname]
            except Exception:
                _mongo_coll = None
        if _mongo_coll is None:
            raise HTTPException(status_code=501, detail="MongoDB not configured")

    # Build record
    rec = {
        "session_id": payload.get("session_id"),
        "timestamp": payload.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "emotion_state": payload.get("emotion"),
        "stress_score": payload.get("stress"),
        "attention_score": payload.get("attention"),
    }
    if not rec["session_id"]:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        _mongo_coll.insert_one(rec)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mongo insert error: {e}")


if __name__ == "__main__":
    uvicorn.run("src.inference.infer_service:app", host="127.0.0.1", port=5001, reload=False)
