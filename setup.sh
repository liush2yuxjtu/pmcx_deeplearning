#!/bin/bash

# IXI MCX 2025 Setup Script
# This script helps set up the environment for running MCX simulations

echo "IXI MCX 2025 Setup Script"
echo "========================"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This setup script is designed for Linux systems."
fi

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "Warning: No NVIDIA GPU detected. MCX simulations require CUDA-compatible GPU."
    echo "You can still run data processing and model training on CPU."
fi

# Check for Python
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "Found ${PYTHON_VERSION}"
else
    echo "Error: Python 3 not found. Please install Python 3.7 or later."
    exit 1
fi

# Check for pip
echo "Checking for pip..."
if command -v pip3 &> /dev/null; then
    echo "pip3 found"
else
    echo "Installing pip..."
    sudo apt update
    sudo apt install python3-pip -y
fi

# Install required packages
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check for MCX
echo "Checking for MCX..."
if python3 -c "import pmcx" &> /dev/null; then
    echo "MCX Python bindings found"
else
    echo "Installing MCX Python bindings..."
    pip3 install pmcx
fi

# Download IXI sample data (if needed)
echo "Setting up data directories..."
mkdir -p ixi_mcx_2025_latent
mkdir -p ixi_mcx_2025_lowres
mkdir -p checkpoints_latent_simple2full
mkdir -p checkpoints_lowres_simple2full
mkdir -p results_latent_simple2full
mkdir -p results_lowres_simple2full

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run './check.sh' to verify system resources"
echo "2. Run 'python 2.pmcx.run.both.py' to execute MCX simulations"
echo "3. Explore Jupyter notebooks for data processing and training"

echo ""
echo "Note: For full IXI dataset processing, you'll need to obtain the IXI dataset"
echo "and process it through the SimNIBS pipeline to generate tissue segmentations."