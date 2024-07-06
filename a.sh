#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1 # request a GPU
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G      
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --account=def-mushrifs
#SBATCH --mail-user=<aksharasoman@gmail.com>
#SBATCH --mail-type=ALL

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install -r requirements.txt --no-index

echo "starting training..."
python main_DCGAN_training.py
