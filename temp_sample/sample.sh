#!/bin/bash

# End-to-end sampling script for IXI MCX 2025 project

# Ensure we're in the project root directory
cd "$(dirname "$(dirname "$0")")"

echo "Starting IXI MCX 2025 sampling from latent..."
echo "Output directory: outputs_latent"

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
mkdir -p outputs_latent

# Run the sampling script
echo "Running sampling script..."
python3 temp_sample/sample_from_latent.py

# Check if sampling completed successfully
if [ $? -eq 0 ]; then
    echo "Sampling completed successfully!"
    echo "Generated NIfTI files are in: outputs_latent"
    echo "Files:"
    echo "  - outputs_latent/generated_sample.nii.gz"
    echo "  - outputs_latent/target.nii.gz"
    echo "  - outputs_latent/input.nii.gz"
else
    echo "Error: Sampling failed"
    exit 1
fi

echo "Done!"
