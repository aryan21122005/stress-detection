import torch
import pytest
from src.models.multihead_cnn import MultiHeadCNN


def test_model_initialization():
    """Test that the model initializes correctly."""
    model = MultiHeadCNN()
    assert model is not None
    
    # Test model output shapes
    dummy_input = torch.randn(1, 3, 224, 224)
    emo, eng, stress = model(dummy_input)
    
    assert emo.shape == (1, 7)    # 7 emotion classes
    assert eng.shape == (1, 4)    # 4 engagement levels
    assert stress.shape == (1, 4) # 4 stress levels


def test_model_load_weights():
    """Test that model can load weights correctly."""
    model = MultiHeadCNN()
    try:
        # Create a dummy state dict
        dummy_state = {'model_state_dict': model.state_dict()}
        torch.save(dummy_state, 'dummy_weights.pth')
        
        # Test loading
        model.load_state_dict(torch.load('dummy_weights.pth')['model_state_dict'])
        assert True
    except Exception as e:
        assert False, f"Failed to load weights: {e}"
