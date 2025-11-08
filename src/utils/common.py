import torch
import torch.nn.functional as F


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ENGAGEMENT_LABELS = ["very_low", "low", "high", "very_high"]
STRESS_LABELS = ["very_low", "low", "high", "very_high"]


def softmax_probs(logits):
    return F.softmax(logits, dim=1)


def top1_label(probs, labels):
    idx = int(probs.argmax(dim=1).item())
    return labels[idx], float(probs[0, idx].item())


def focus_score_from_engagement(probs):
    w = torch.tensor([0.1, 0.4, 0.7, 0.95], dtype=probs.dtype, device=probs.device)
    s = (probs * w).sum(dim=1)
    return float(s.item())


def stress_score_from_frustration(probs, emotion_confidence=1.0):
    """
    Calculate stress score from stress level probabilities.
    Args:
        probs: Tensor of shape [batch_size, 4] containing probabilities for [No Stress, Low, Medium, High]
        emotion_confidence: Confidence of the emotion prediction (0-1)
    """
    # More conservative weights - higher threshold for stress
    w = torch.tensor([0.0, 0.2, 0.5, 0.8], 
                    dtype=probs.dtype, 
                    device=probs.device)
    
    # Only consider stress if confidence is high enough
    if probs[0, 0] > 0.7:  # If very confident about 'No Stress'
        w = torch.tensor([0.0, 0.1, 0.2, 0.3], 
                        dtype=probs.dtype, 
                        device=probs.device)
    
    # Calculate weighted score
    score = (probs * w).sum(dim=1)
    
    # Adjust by emotion confidence
    score = score * (0.5 + 0.5 * emotion_confidence)
    
    # Apply non-linearity to make mid-range values more common
    score = score ** 0.8
    
    return float(score.item())
