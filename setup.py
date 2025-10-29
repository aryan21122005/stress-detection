from setuptools import setup, find_packages

setup(
    name="stress-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0.76',
        'numpy>=1.24.0',
        'pillow>=10.0.0',
        'tqdm>=4.66.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'pyyaml>=6.0.0',
        'mediapipe>=0.10.0',
        'fastapi>=0.95.0',
        'uvicorn>=0.22.0',
        'python-multipart>=0.0.6',
    ],
    python_requires='>=3.8',
)
