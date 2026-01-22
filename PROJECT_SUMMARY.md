# IXI MCX 2025 Project Summary

## Project Status

This repository contains a comprehensive computational framework for simulating near-infrared light transport through human head tissues using Monte Carlo eXtreme (MCX) methods. The project focuses on transcranial photobiomodulation (tPBM) applications using the IXI dataset of MRI scans.

## Key Accomplishments

1. **Comprehensive Documentation Created**:
   - Detailed `QWEN.md` technical documentation
   - User-friendly `README.md` overview
   - Dependency list in `requirements.txt`
   - Setup scripts for easy deployment

2. **Core Infrastructure Established**:
   - MCX simulation pipeline with `2.pmcx.run.both.py`
   - Utility modules in `pmcx_utils/` for reusable functionality
   - Data processing notebooks for training deep learning models
   - System resource checking with `check.sh`

3. **Deep Learning Integration**:
   - Latent space modeling of MCX simulations
   - Multiple training approaches (lowres, latent)
   - Conditional diffusion models using Rectified Flow
   - Model checkpointing and result organization

## Project Architecture

### Simulation Pipeline
1. **Stage 1**: Source positioning using EEG coordinates
2. **Stage 2**: MCX light transport simulation in two modes:
   - Simple mode: Rapid prototyping (~1e6 photons)
   - Full mode: High accuracy (~1e10 photons)

### Deep Learning Components
- **Latent Space Training**: Compressed representation learning
- **Multi-resolution Models**: Lowres and latent space variants
- **Diffusion Models**: Rectified Flow for deterministic generation

### Data Organization
- Hierarchical structure for different resolution levels
- Model checkpoints separated by configuration
- Results organized by simulation type
- Consistent naming conventions

## Technical Highlights

### GPU-Accelerated Computing
- CUDA-accelerated Monte Carlo simulations
- Efficient memory management for large datasets
- Parallel processing of multiple subjects

### Medical Image Processing
- NIfTI format handling for neuroimaging data
- Tissue segmentation with 6 distinct tissue types
- Anatomically accurate head models from IXI dataset

### Scientific Accuracy
- Physically accurate optical properties for each tissue
- Proper treatment of absorption and scattering coefficients
- Validation against analytical solutions where possible

## Next Steps for Users

1. **System Setup**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Resource Checking**:
   ```bash
   ./check.sh
   ```

3. **Running Simulations**:
   ```bash
   python 2.pmcx.run.both.py
   ```

4. **Training Models**:
   - Use Jupyter notebooks (`1.data.*.ipynb`)
   - Start with lowres training for faster iteration
   - Progress to latent space for higher quality

## Potential Applications

### Clinical Applications
- Optimizing tPBM treatment protocols
- Personalized brain stimulation planning
- Treatment outcome prediction

### Research Applications
- Light-tissue interaction studies
- Novel electrode montage investigation
- Population-level variation analysis

## Repository Status

✅ **Ready for use** - All core infrastructure is in place
✅ **Well documented** - Comprehensive documentation available
✅ **Modular design** - Easy to extend and modify
✅ **Production ready** - Used in active research projects

## Recommendations for New Users

1. Start with the `README.md` for overview
2. Review `QWEN.md` for technical details
3. Check system requirements with `./check.sh`
4. Run sample simulations with `2.pmcx.run.both.py`
5. Explore Jupyter notebooks for data processing workflows
6. Begin training with lowres models for faster iteration