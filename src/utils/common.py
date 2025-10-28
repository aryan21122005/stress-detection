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


def stress_score_from_frustration(probs):
    w = torch.tensor([0.1, 0.4, 0.7, 0.95], dtype=probs.dtype, device=probs.device)
    s = (probs * w).sum(dim=1)
    return float(s.item())
