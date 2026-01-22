# IXI MCX Minimal Implementation

This is a minimal implementation of the IXI MCX deep learning project for running Monte Carlo eXtreme (MCX) simulations for computational modeling of light transport in biological tissues.

## Prerequisites

- Python 3.7+
- NVIDIA GPU compatible with CUDA
- MCX simulation toolkit
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
   \`\`\`
   git clone <repository_url>
   cd ixi_mcx_minimal
   \`\`\`

2. Install required packages:
   \`\`\`
   pip install -r requirements.txt
   # Or install with pip: 
   pip install numpy scipy matplotlib nibabel pandas pmcx torch monai rectified-flow-pytorch
   \`\`\`

## Usage

1. Prepare your data:
   - Ensure you have a NIfTI (.nii.gz) file with tissue segmentation
   - Ensure you have an EEG electrode position CSV file

2. Run the minimal implementation:
   \`\`\`
   python run_minimal.py
   \`\`\`

   Or specify custom data paths:
   \`\`\`
   DATA_PATH="/path/to/your/data.nii.gz" SUBJECT_CSV="/path/to/your/eeg_positions.csv" python run_minimal.py
   \`\`\`

## Configuration

The script runs in "simple" mode by default, which uses fewer photons for faster execution.
To modify parameters, edit run_minimal.py directly.

## Project Structure

- \`run_minimal.py\`: Main entry point for running the minimal implementation
- \`ixi_mcx_minimal/pmcx_utils/\`: Core utilities for MCX simulations
- \`requirements.txt\`: Python dependencies
- \`setup.py\`: Package setup configuration

## Notes

- This is a minimal implementation; for full functionality, see the original repository
- The MCX simulation requires a GPU to run effectively
- Default parameters use 1e6 photons in simple mode for faster execution

