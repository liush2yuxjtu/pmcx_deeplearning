# IXI MCX 2025 Project Documentation

## Project Overview

This project involves Monte Carlo eXtreme (MCX) simulations for computational modeling of light transport in biological tissues, particularly focusing on transcranial photobiomodulation (tPBM) applications. The project uses the IXI dataset (a database of MRI scans) to create realistic head models for simulating light propagation through brain tissues.

The main objective is to simulate how near-infrared light penetrates through different head tissues (scalp, skull, CSF, gray matter, white matter) to reach specific brain regions, which is crucial for optimizing tPBM therapy protocols.

## Key Components

### Data Structure
- **IXI Dataset**: MRI scans used to create anatomically accurate head models
- **Segmentation Files**: Tissue classification maps (final_tissues.nii.gz) identifying different tissue types
- **MCX Simulation Outputs**: Forward and inverse MCX solutions stored as NIfTI files
- **Electrode Positions**: EEG electrode coordinates (e.g., Fpz, Fp1) for light source placement

### Main Technologies
- **Python**: Primary programming language
- **MCX (Monte Carlo eXtreme)**: GPU-accelerated Monte Carlo simulation engine
- **NIfTI Processing**: Neuroimaging data format handling
- **PyTorch**: Deep learning framework for latent space modeling
- **MONAI**: Medical Open Network for AI for medical image processing

### Key Files and Directories

#### Core Scripts
- `2.pmcx.run.both.py`: Main execution script for running MCX simulations in both simple and full modes
- `pmcx.ipynb`: Jupyter notebook for MCX simulation workflows
- Data processing notebooks (`1.data.*.ipynb`) for training data preparation
- Utility modules in `pmcx_utils/` containing core functionality

#### Data Directories
- `ixi_mcx_2025_latent/`, `ixi_mcx_2025_lowres/`: Processed simulation data at different resolutions
- `checkpoints_*`: Model checkpoints for trained neural networks
- `results_*`: Simulation outputs and intermediate results
- `models/`: Pre-trained models

#### Configuration Files
- `file_dicts.json`: File mapping dictionary for data organization
- `check.sh`: System resource checking script for GPU/CPU availability

## Project Workflow

### 1. Data Preparation
The project processes IXI dataset MRI scans to create tissue-segmented head models:
- High-resolution tissue segmentation using SimNIBS pipeline
- Conversion to MCX-compatible format
- Creation of electrode positioning system based on EEG coordinates

### 2. MCX Simulation Pipeline
Two-stage simulation approach:

#### Stage 1: Source Positioning
- Load subject MRI data and tissue segmentation
- Extract electrode coordinates from EEG positioning CSV
- Transform coordinates to voxel space
- Calculate optimal light source direction based on target region

#### Stage 2: Light Transport Simulation
Two modes of operation:
- **Simple Mode**: Lower resolution simulation for rapid prototyping (1e6 photons)
- **Full Mode**: High-resolution simulation for accurate results (~1e10 photons)

Parameters:
- Wavelength: 1064nm (near-infrared)
- Power: 250mW
- Exposure Time: 8 minutes
- Tissue optical properties predefined for each tissue type

### 3. Deep Learning Integration
- Latent space modeling of MCX simulation results
- Training diffusion models to predict full simulations from simple inputs
- Two resolution levels: lowres and latent space representations

## Building and Running

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (NVIDIA RTX recommended)
- MCX simulation toolkit
- PyTorch 1.9+
- Required Python packages:
  - nibabel (NIfTI file handling)
  - numpy, scipy (numerical computing)
  - matplotlib (visualization)
  - pandas (data processing)
  - pmcx (Monte Carlo eXtreme Python interface)
  - monai (medical image processing)
  - rectified-flow-pytorch (diffusion modeling)

### Setup Instructions
```bash
# Check system resources and available nodes
./check.sh

# Install required packages (if not already installed)
pip install pmcx nibabel torch monai pandas

# Clone and setup MCX utilities
# (MCX binaries need to be compiled separately)
```

### Execution
```bash
# Run main simulation pipeline
python 2.pmcx.run.both.py

# Train deep learning models
# Using notebooks like 1.data.lowres.train.v2.ipynb

# Check system resources before running
./check.sh
```

## System Architecture

### MCX Simulation Core
The simulation uses physically accurate light transport modeling:
- **Photon Physics**: Proper treatment of absorption and scattering coefficients
- **Tissue Modeling**: Six distinct tissue types with literature-derived optical properties
- **GPU Acceleration**: CUDA-accelerated Monte Carlo simulation for performance

### Deep Learning Pipeline
- **Latent Space Encoding**: Dimensionality reduction of simulation results
- **Conditional Diffusion Models**: Rectified Flow for deterministic generation
- **Multi-scale Processing**: Low-resolution and latent space training variants

### Data Organization
- Hierarchical subject/electrode organization
- Paired simple/full simulation datasets
- Consistent NIfTI format for interoperability

## Development Notes

### Code Structure
```
ixi_mcx_2025/
├── pmcx_utils/           # Core MCX utilities
│   ├── __init__.py
│   └── core.py           # Stage 1 & 2 implementations
├── notebooks/            # Jupyter notebooks for exploration
├── data/                 # Processed datasets
├── models/               # Trained models
└── results/              # Simulation outputs
```

### Key Functions
1. `run_stage1()`: Coordinate transformation and source positioning
2. `run_stage2()`: MCX simulation execution and result processing
3. `show_src_vol()`: Visualization of source positions
4. `view_result()`: 3D visualization of simulation results

### Directory Structure Considerations
- Large datasets organized by resolution level (lowres, latent)
- Model checkpoints separated by training configuration
- Outputs saved in structured directories for reproducibility
- Temporary files managed in `pmcx_temp_data/`

## Performance Optimization

### GPU Computing
- MCX leverages CUDA for massive parallelism
- Approximately 10^10 photons simulated for full accuracy
- Single simulation takes minutes on high-end GPUs

### Memory Management
- Efficient NIfTI loading with memory mapping
- Progressive zooming for resolution matching
- Logarithmic scaling for numerical stability

## Troubleshooting

### Common Issues
1. **CUDA Memory Errors**: Reduce simulation resolution or batch size
2. **File Path Issues**: Ensure all data directories are correctly linked
3. **GPU Availability**: Use `check.sh` to verify GPU status before running
4. **MCX Installation**: Verify MCX binaries are properly compiled and accessible

### Resource Monitoring
```bash
# Monitor GPU usage
nvidia-smi

# Check disk space
df -h

# Check cluster node status
./check.sh
```

## Contributing

### Code Standards
- Follow PEP 8 Python style guidelines
- Use meaningful variable names and comments
- Maintain consistent directory structure
- Document all parameters and configurations

### Testing
- Validate simulation outputs against expected photon counts
- Verify tissue segmentation accuracy
- Cross-check results with analytical solutions where possible
- Test on multiple IXI subjects for generalization

## Scientific Background

This project implements computational modeling of transcranial photobiomodulation (tPBM), a non-invasive brain stimulation technique using near-infrared light. The physics of light transport in biological tissues requires solving the radiative transfer equation, which is computationally intensive. 

MCX provides a stochastic solution using Monte Carlo methods, tracking billions of photon packets through tissue volumes with appropriate absorption and scattering coefficients. The resulting fluence distributions inform optimal placement of light sources for targeting specific brain regions.

The deep learning component learns a mapping from computationally inexpensive simple simulations to accurate full simulations, potentially enabling real-time treatment planning.

## References

1. Fang, Q., & Boas, D. A. (2009). Monte Carlo simulation of photon migration in 3D turbid media accelerated by graphics processing units. Optics express, 17(22), 20178-20190.

2. Gemert, M. J., Star, W. M., & Faber, D. J. (1989). Mathematical model for grating coupler theory applied to the determination of the optical properties of human brain at radio and microwave frequencies. Lasers in surgery and medicine, 9(1), 57-65.

3. Scholkmann, F., Kleiser, S., Metz, A. J., Zimmermann, R., Mata Pires, D., Wolf, U., ... & Wolf, M. (2014). A review on continuous wave functional near-infrared spectroscopy and imaging instrumentation and methodology. Neuroimage, 85, 6-27.