import torch
from torch.utils.data import DataLoader
from src.datasets.fer2013 import FER2013Dataset
from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_val_transform
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    model = MultiHeadCNN()
    checkpoint = torch.load("checkpoints/fer2013_best.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load test dataset
    transform = get_val_transform(224)  # Assuming image size 48 as used in training
    test_ds = FER2013Dataset("data/fer2013/fer2013.csv", usage="PrivateTest", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs, _, _ = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    evaluate()
