#!/bin/bash

#SBATCH --job-name=EVAL_CLS_dito_3d_medmnist_standalone_v2_copy.sh          # Job name
#SBATCH --output=logs/EVAL_CLS_dito_3d_medmnist_standalone_v2_copy.sh%j.out           # Standard output log (%j = job ID)
#SBATCH --error=logs/EVAL_CLS_dito_3d_medmnist_standalone_v2_copy.sh%j.err            # Standard error log (%j = job ID)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of tasks per node
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task
#SBATCH --gres=gpu:1                         # Number of GPUs (format: gpu:type:count)
#SBATCH --mem=12G                            # Memory per node
#SBATCH --nodelist=node[2,3]

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

source /data/software/miniconda/etc/profile.d/conda.sh
conda activate liushiyu

# Verify GPU availability
echo "GPU Information:"
nvidia-smi
echo "========================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH="/data/home/syliu/.conda/envs/liushiyu/bin/python"

export FOLDER_NAME="/data4/brats25/dito_2d3d_demo/demos/exp_evaluation_tasks/ADNI"
export SCRIPT_NAME="EVAL_CLS_dito_3d_medmnist_standalone_v2_copy.py"
# Create necessary directories
mkdir -p logs
mkdir -p output
mkdir -p ckpt

# Change to project directory
cd $FOLDER_NAME

# Run your training script
echo "Starting training..."
python $SCRIPT_NAME

# Print completion information
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $?"

# Optional: Copy results to backup location
# cp -r output /backup/location/
# cp -r ckpt /backup/location/

exit 0