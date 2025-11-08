import torch
from src.training.train_daisee import train as train_daisee
from src.training.train_fer2013 import train as train_fer2013
import argparse
import os

def main():
    # Common arguments
    parser = argparse.ArgumentParser(description='Sequential training on DAiSEE and FER2013')
    
    # DAiSEE arguments
    parser.add_argument('--daisee-train', type=str, default='data/daisee/train.csv',
                      help='Path to DAiSEE training CSV')
    parser.add_argument('--daisee-val', type=str, default='data/daisee/val.csv',
                      help='Path to DAiSEE validation CSV')
    
    # FER2013 arguments
    parser.add_argument('--fer2013-csv', type=str, default='data/fer2013/fer2013.csv',
                      help='Path to FER2013 CSV')
    
    # Common training arguments
    parser.add_argument('--out', type=str, default='checkpoints',
                      help='Output directory for checkpoints')
    parser.add_argument('--batch-size-daisee', type=int, default=16,
                      help='Batch size for DAiSEE training')
    parser.add_argument('--batch-size-fer2013', type=int, default=32,
                      help='Batch size for FER2013 training')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Learning rate')
    parser.add_argument('--workers', type=int, default=2,
                      help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    
    # Phase 1: Train on DAiSEE
    print("\n" + "="*50)
    print("Starting DAiSEE Training (5 epochs)")
    print("="*50)
    
    daisee_args = argparse.Namespace(
        manifest=args.daisee_train,
        val_manifest=args.daisee_val,
        batch_size=args.batch_size_daisee,
        epochs=5,
        lr=args.lr,
        out=args.out,
        weights='',  # Start from scratch
        image_size=112,
        workers=args.workers
    )
    
    train_daisee(daisee_args)
    
    # Phase 2: Train on FER2013
    print("\n" + "="*50)
    print("Starting FER2013 Training (5 epochs)")
    print("="*50)
    
    # Load the best DAiSEE model if available
    daisee_weights = os.path.join(args.out, 'daisee_best.pth')
    
    fer2013_args = argparse.Namespace(
        csv=args.fer2013_csv,
        batch_size=args.batch_size_fer2013,
        epochs=5,
        lr=args.lr,
        out=args.out,
        weights=daisee_weights if os.path.exists(daisee_weights) else '',
        image_size=48,
        workers=args.workers
    )
    
    train_fer2013(fer2013_args)
    
    # Save the final model with a new name
    final_model_path = os.path.join(args.out, 'fer2013_sequential.pth')
    
    if os.path.exists(final_model_path):
        import shutil
        # Save the model state directly
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 5,
            'best_acc': 0.0  # Will be updated during training
        }, final_model_path)
        print(f"\nModel saved to {final_model_path}")
    else:
        print("\nWarning: Could not find the final model to copy.")

if __name__ == "__main__":
    main()
