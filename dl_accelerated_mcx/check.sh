#!/bin/bash

# Check system resources and environment for DL Accelerated MCX Simulation

echo "DL Accelerated MCX Simulation - System Check"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "$python_version"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
    echo ""
    
    # Check CUDA version
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"
        echo ""
    else
        echo "PyTorch not installed or CUDA not available"
        echo ""
    fi
else
    echo "No NVIDIA GPU detected or nvidia-smi not available"
    echo ""
fi

# Check available disk space
echo "Checking disk space..."
df -h .
echo ""

# Check memory usage
echo "Checking memory usage..."
free -h
echo ""

# Check CPU info
echo "Checking CPU info..."
lscpu | grep -E "(Architecture|CPU\(s\)|Model name|Thread\(s\) per core|Core\(s\) per socket)"
echo ""

# Check Python packages
echo "Checking Python packages..."
required_packages=("torch" "numpy" "scipy" "nibabel" "monai" "matplotlib" "pandas" "tqdm" "accelerate" "ema-pytorch" "rectified-flow-pytorch")
echo "Required packages:"
for package in "${required_packages[@]}"; do
    if python -c "import $package; print('$package: OK')" 2>/dev/null; then
        echo "  $package: OK"
    else
        echo "  $package: MISSING"
    fi
done
echo ""

# Check MCX installation
echo "Checking MCX installation..."
if python -c "import pmcx; print('PMCX: OK')" 2>/dev/null; then
    echo "  PMCX: OK"
else
    echo "  PMCX: MISSING"
fi
echo ""

echo "System check completed!"