#!/bin/bash
# Setup script for running TensorFlow with GPU in WSL2

set -e  # Exit on error

echo "================================================"
echo "🚀 Setting up TensorFlow GPU in WSL2"
echo "================================================"

# Get the Windows path and convert to WSL path
WIN_PATH="/mnt/c/Lokal_Code/Mitgliedschafts Voraussage/Mitgliedschafts Voraussage"
cd "$WIN_PATH" || exit 1

echo "✅ Working directory: $(pwd)"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi

echo "✅ Python version: $(python3 --version)"
echo ""

# Create WSL virtual environment if it doesn't exist
if [ ! -d "venv_wsl" ]; then
    echo "📦 Creating WSL virtual environment..."
    python3 -m venv venv_wsl
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_wsl/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install main requirements
echo "📦 Installing main requirements..."
pip install -r requirements.txt

# Install TensorFlow with CUDA support
echo "🎯 Installing TensorFlow with CUDA support..."
pip install tensorflow[and-cuda]

# Install DNN requirements
echo "📦 Installing DNN requirements..."
pip install scikeras cloudpickle

echo ""
echo "================================================"
echo "🎉 Setup Complete!"
echo "================================================"
echo ""

# Test GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'Number of GPUs detected: {len(gpus)}')
if gpus:
    for gpu in gpus:
        print(f'  - {gpu}')
    print('✅ GPU DETECTED! Ready for CUDA-accelerated training!')
else:
    print('⚠️  No GPU detected. Please ensure NVIDIA drivers are installed.')
"

echo ""
echo "================================================"
echo "🚀 Ready to train! Run:"
echo "   python3 training_lead_generation_model.py --backend dnn_gpu"
echo "================================================"
