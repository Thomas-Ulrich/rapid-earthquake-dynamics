#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J 2015_1.1
#Output and error (also --output, --error):
#SBATCH -o ./%j.%x.out
#SBATCH -e ./%j.%x.out

#Initial working directory:
#SBATCH --chdir=./

#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=thomas.ulrich@lmu.de
#SBATCH --no-requeue
#SBATCH --export=ALL
#SBATCH --account=pn49ha
#SBATCH --ntasks-per-node=1
#EAR may impact code performance
#SBATCH --ear=off
##SBATCH --nodes=20 --partition=general --time=02:30:00
#SBATCH --nodes=1 --partition=test --time=00:30:00 
source /etc/profile.d/modules.sh

module load slurm_setup

#Run the program:
export MP_SINGLE_THREAD=yes
unset KMP_AFFINITY
export OMP_NUM_THREADS=48
export OMP_PLACES="cores(48)"

#export MP_SINGLE_THREAD=no
#export OMP_NUM_THREADS=1
#export MP_TASK_AFFINITY=core:$OMP_NUM_THREADS


echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

module load seissol/before1168-intel23-o4-elas-dunav-single-impi


srun -u python ../rapid-earthquake-dynamics/step2_generate_dr_input_files.py
