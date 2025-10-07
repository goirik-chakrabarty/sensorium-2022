#!/bin/bash

#SBATCH --job-name=cl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=grete:interactive    # see https://www.hlrn.de/doc/display/PUB/Compute+node+partitions for a list of accessible hlrn partitions
#SBATCH -G 1g.20gb:1                   # number of gpus per node
#SBATCH --time 6:00:00
#SBATCH -C inet 
#SBATCH -A agsinz_foundation_model_brain
#SBATCH --mem=50G
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de

export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer
module load cuda/12.2.1
module load miniforge3
module load git

cd /mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/curriculum-learning/sensorium-2022

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
#nvcc -V

# Define variables
PROJECT_ROOT="$PWD"
SIF_IMAGE="$PWD/sensorium.sif"
ENV_FILE=".env"

# Run the container
apptainer exec \
 --nv \
 --bind "$PROJECT_ROOT:/project" \
 --bind "$PROJECT_ROOT/notebooks:/notebooks" \
 "$SIF_IMAGE" \
 bash apptainer-runners/run_debugger.sh