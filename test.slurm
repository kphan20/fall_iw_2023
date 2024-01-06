#!/bin/bash
#SBATCH --job-name=cos-iw-test     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-2
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=kphan@princeton.edu

module purge

source env/bin/activate

python test.py