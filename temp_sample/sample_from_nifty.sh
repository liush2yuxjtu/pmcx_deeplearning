#!/bin/bash

# End-to-end sampling script for IXI MCX 2025 project - NIfTI input

# Ensure we're in the project root directory
cd "$(dirname "$(dirname "$0")")"

echo "Starting IXI MCX 2025 sampling from NIfTI..."
echo "Output directory: temp_sample/outputs"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if required directories exist
if [ ! -d "models" ]; then
    echo "Error: models/ directory not found"
    exit 1
fi

if [ ! -d "checkpoints_latent_simple2full" ]; then
    echo "Error: checkpoints_latent_simple2full/ directory not found"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p temp_sample/outputs

# Run the sampling script with debug flag if provided
echo "Running sampling script..."
python3 temp_sample/sample_from_nifty.py "$@"

# Check if sampling completed successfully
if [ $? -eq 0 ]; then
    echo "Sampling completed successfully!"
    echo "Generated NIfTI files are in: temp_sample/outputs"
    echo "Files:"
    echo "  - temp_sample/outputs/generated_sample.nii.gz"
    echo "  - temp_sample/outputs/target.nii.gz"
    echo "  - temp_sample/outputs/input.nii.gz"
else
    echo "Error: Sampling failed"
    exit 1
fi

echo "Done!"