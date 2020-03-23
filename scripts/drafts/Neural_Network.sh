#!/bin/bash
#SBATCH -A p31049               # Allocation
#SBATCH -p short                # Queue
#SBATCH -t 02:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --mem=10G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=1     # Number of Cores (Processors)
#SBATCH --mail-user=,meghutch@u.northwestern.edu  # Designate email address for job communications
#SBATCH --mail-type=FAIL     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --job-name="Neural_Network"       # Name of job 


# unload any modules that carried over from your command line session
module purge

# load modules you need to use
module load python/anaconda3.6
module load java

source activate envs_dl

# A command you actually want to execute:

python Neural_Network.py > /projects/p31049/Multi_Cancer_DL/04_Results/HL100_50_25_LR_001_E100.out
