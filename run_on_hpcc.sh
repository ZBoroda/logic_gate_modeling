#!/bin/sh

#SBATCH --account=guest
#SBATCH --partition=guest-compute
#SBATCH --qos=low
#SBATCH --time=24:00:00
#SBATCH --job-name=run_simulation
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zacharyboroda@brandeis.edu
#SBATCH --output=output_mutation_depends_on_size.txt
#SBATCH --tasks=1
#SBATCH --cpus-per-task=128

module load share_modules/ANACONDA/5.3_py3

srun python main.py