import os
import sys
import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Common test fixtures can be defined here

@pytest.fixture
def sample_image():
    """Generate a sample image tensor for testing."""
    import torch
    return torch.rand(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224

@pytest.fixture
def model():
    """Fixture that provides an initialized model."""
    from src.models.multihead_cnn import MultiHeadCNN
    return MultiHeadCNN().eval()
