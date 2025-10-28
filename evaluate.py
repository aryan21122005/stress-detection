import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.datasets.fer2013 import FER2013Dataset
from src.datasets.daisee import DAiSEEDataset
from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_val_transform

def evaluate_model(model, data_loader, device, model_type='fer2013'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if model_type == 'fer2013':
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs, _, _ = model(x)
                _, predicted = torch.max(outputs.data, 1)
            else:  # daisee
                x, (y_eng, y_stress) = batch
                x, y_eng = x.to(device), y_eng.to(device)
                _, eng_outputs, _ = model(x)
                _, predicted = torch.max(eng_outputs.data, 1)
                y = y_eng  # For engagement prediction
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['fer2013', 'daisee'], 
                       help='Type of model to evaluate')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers for data loading')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = MultiHeadCNN()
    checkpoint = torch.load(args.weights, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    
    # Load dataset
    if args.model_type == 'fer2013':
        test_dataset = FER2013Dataset(
            'data/fer2013/fer2013.csv', 
            usage='PublicTest',  # Using PublicTest as validation set
            transform=get_val_transform(224)
        )
    else:  # daisee
        test_dataset = DAiSEEDataset(
            'data/DAiSEE/val.csv',  # Make sure this path is correct
            transform=get_val_transform(224)
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Evaluate
    accuracy = evaluate_model(model, test_loader, device, args.model_type)
    print(f"\n{args.model_type.upper()} Model Evaluation")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Print class-wise accuracy if needed
    if args.model_type == 'fer2013':
        print("\nClass-wise accuracy:")
        classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        class_correct = [0] * len(classes)
        class_total = [0] * len(classes)
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs, _, _ = model(x)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == y).squeeze()
                
                for i in range(len(y)):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        for i in range(len(classes)):
            if class_total[i] > 0:
                print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    return accuracy

if __name__ == '__main__':
    main()
