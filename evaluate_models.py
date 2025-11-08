import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

# Import datasets and model
from src.datasets.fer2013 import FER2013Dataset
from src.datasets.daisee import DAiSEEDataset
from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_val_transform

def evaluate_fer2013():
    print("\n" + "="*50)
    print("Evaluating FER2013 Model")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = MultiHeadCNN(num_emotions=7)  # For FER2013
    try:
        checkpoint = torch.load("checkpoints/fer2013_best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded FER2013 model from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"Error loading FER2013 model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load test data
    transform = get_val_transform(224)
    test_ds = FER2013Dataset("data/fer2013/fer2013.csv", usage="PrivateTest", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating FER2013"):
            images = images.to(device)
            emo_logits, _, _ = model(images)
            _, preds = torch.max(emo_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print("\nFER2013 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"FER2013 Overall Accuracy: {accuracy*100:.2f}%")

def evaluate_daisee():
    print("\n" + "="*50)
    print("Evaluating DAiSEE Model")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = MultiHeadCNN(num_emotions=7, num_engagement=4, num_stress=4)  # For DAiSEE
    try:
        checkpoint = torch.load("checkpoints/daisee_best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded DAiSEE model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best average accuracy: {checkpoint.get('best_avg_acc', 0)*100:.2f}%")
    except Exception as e:
        print(f"Error loading DAiSEE model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load validation data
    transform = get_val_transform(112)
    val_ds = DAiSEEDataset("data/daisee/val.csv", transform=transform)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    # Evaluate
    all_e_preds = []
    all_e_labels = []
    all_s_preds = []
    all_s_labels = []
    
    with torch.no_grad():
        for x, e, s in tqdm(val_loader, desc="Evaluating DAiSEE"):
            x = x.to(device)
            _, eng_logits, stress_logits = model(x)
            e_pred = eng_logits.argmax(dim=1)
            s_pred = stress_logits.argmax(dim=1)
            all_e_preds.extend(e_pred.cpu().numpy())
            all_e_labels.extend(e.numpy())
            all_s_preds.extend(s_pred.cpu().numpy())
            all_s_labels.extend(s.numpy())
    
    # Calculate metrics
    print("\nDAiSEE Engagement Classification Report:")
    print(classification_report(all_e_labels, all_e_preds, digits=4))
    e_accuracy = (np.array(all_e_preds) == np.array(all_e_labels)).mean()
    print(f"Engagement Accuracy: {e_accuracy*100:.2f}%")
    
    print("\nDAiSEE Stress Classification Report:")
    print(classification_report(all_s_labels, all_s_preds, digits=4))
    s_accuracy = (np.array(all_s_preds) == np.array(all_s_labels)).mean()
    print(f"Stress Accuracy: {s_accuracy*100:.2f}%")
    
    avg_accuracy = (e_accuracy + s_accuracy) / 2
    print(f"\nDAiSEE Average Accuracy: {avg_accuracy*100:.2f}%")

if __name__ == "__main__":
    evaluate_fer2013()
    evaluate_daisee()
