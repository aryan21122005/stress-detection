import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_inference_transform
from src.utils.face_detection import detect_largest_face_bgr, crop_face_square
from src.utils.common import EMOTION_LABELS, STRESS_LABELS, softmax_probs, top1_label, stress_score_from_frustration
from src.utils.gaze_tracker import gaze_tracker
from src.utils.suggestions import suggest
from src.utils.data_logger import session_logger


def load_model(weights, device):
    model = MultiHeadCNN()
    model.to(device)
    if weights:
        ckpt = torch.load(weights, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception:
                pass
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224, help="Input image size (224x224 is recommended for better accuracy)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    tfm = get_inference_transform(args.image_size)
    # Start a new logging session
    session_id = session_logger.start_session()
    print(f"Started new session: {session_id}")
    
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        session_logger.save_session()
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox = detect_largest_face_bgr(frame)
        if bbox is None:
            disp = frame.copy()
            cv2.putText(disp, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Stress Detection", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save session
                saved_file = session_logger.save_session()
                print(f"\nSession saved to: {saved_file}")
                # Start a new session
                session_id = session_logger.start_session()
                print(f"Started new session: {session_id}")
            continue
        face = crop_face_square(frame, bbox)
        if face is None or face.size == 0:
            disp = frame.copy()
            cv2.putText(disp, "Face crop failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Stress Detection", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save session
                saved_file = session_logger.save_session()
                print(f"\nSession saved to: {saved_file}")
                # Start a new session
                session_id = session_logger.start_session()
                print(f"Started new session: {session_id}")
            continue
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        x = tfm(im).unsqueeze(0).to(device)
        # Get model predictions
        with torch.no_grad():
            emo_logits, _, stress_logits = model(x)
            probs = torch.softmax(emo_logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        # Get emotion predictions with confidence
        emo_probs = softmax_probs(emo_logits)
        emo_label, emo_conf = top1_label(emo_probs, EMOTION_LABELS)
        emo_conf_pct = int(round(max(0.0, min(1.0, emo_conf)) * 100))
        
        # Get stress predictions
        stress_probs = softmax_probs(stress_logits)
        
        # Adjust stress based on detected emotion
        if emo_conf > 0.6:  # Only adjust if confident about emotion
            if emo_label in ['happy', 'surprise']:
                # Significantly reduce stress for positive emotions
                adjustment = torch.tensor([2.0, 0.5, 0.2, 0.1], 
                                       device=stress_probs.device)
                stress_probs = stress_probs * adjustment
            elif emo_label in ['sad', 'angry', 'disgust', 'fear']:
                # Increase stress for negative emotions
                adjustment = torch.tensor([0.3, 0.8, 1.5, 2.0],
                                       device=stress_probs.device)
                stress_probs = stress_probs * adjustment
            
            # Renormalize to ensure probabilities sum to 1
            stress_probs = stress_probs / stress_probs.sum()
        
        # Calculate final stress score with confidence
        stress_score = stress_score_from_frustration(stress_probs, emotion_confidence=emo_conf)
        stress_pct = int(round(max(0.0, min(1.0, stress_score)) * 100))
        attention_score, debug_info = gaze_tracker.estimate_screen_attention(frame)
        focus_pct = int(round(attention_score * 100))
        
        # Log the data
        session_logger.log_data(
            attention_score=attention_score,
            emotion=emo_label,
            emotion_confidence=emo_conf,
            stress_level=stress_score
        )
        
        # Reset attention score if eyes are closed
        if debug_info and not debug_info.get('eyes_open', True):
            focus_pct = 0
        tips = suggest(emo_label, stress_score, attention_score)
        disp = frame.copy()
        x0, y0, w, h = bbox
        cv2.rectangle(disp, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
        
        # Visual feedback for eye state
        if debug_info and not debug_info.get('eyes_open', True):
            cv2.putText(disp, "Eyes Closed!", (x0, y0 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(disp, f"Emotion: {emo_label} ({emo_conf_pct}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Draw stress level with color coding and progress bar
        stress_color = (0, 255, 0)  # Green
        if stress_pct > 70:
            stress_color = (0, 0, 255)  # Red for high stress
        elif stress_pct > 40:
            stress_color = (0, 165, 255)  # Orange for medium stress
            
        # Draw stress bar with border and filled progress
        bar_width = 200
        bar_height = 20
        bar_x, bar_y = 10, 60
        
        # Background
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress fill
        fill_width = int(bar_width * stress_pct / 100)
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), stress_color, -1)
        # Border
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
        
        # Text with colored background for better visibility
        text = f"Stress: {stress_pct}%"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = bar_x + bar_width + 10
        text_y = bar_y + bar_height - 5
        
        # Text background
        cv2.rectangle(disp, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (40, 40, 40), -1)
        # Text
        cv2.putText(disp, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, stress_color, 2, cv2.LINE_AA)
        # Draw attention score with progress bar
        attention_color = (0, 255, 0) if focus_pct > 70 else (0, 200, 255) if focus_pct > 40 else (0, 100, 255)
        
        # Draw attention bar with border and filled progress
        bar_x, bar_y = 10, 90
        fill_width = int(bar_width * focus_pct / 100)
        
        # Background
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress fill
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), attention_color, -1)
        # Border
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
        
        # Text with colored background for better visibility
        text = f"Attention: {focus_pct}%"
        text_x = bar_x + bar_width + 10
        text_y = bar_y + bar_height - 5
        
        # Text background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(disp, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (40, 40, 40), -1)
        # Text
        cv2.putText(disp, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, attention_color, 2, cv2.LINE_AA)
        if tips:
            cv2.putText(disp, f"Tip: {tips[0]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Stress Detection", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # Save the session before exiting
    saved_file = session_logger.save_session()
    print(f"\nSession saved to: {saved_file}")
    
    cap.release()
    cv2.destroyAllWindows()
    gaze_tracker.release()


if __name__ == "__main__":
    main()
