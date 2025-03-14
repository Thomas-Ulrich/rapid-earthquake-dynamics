#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J gen_syn
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
#SBATCH --nodes=1 --partition=test --time=00:30:00  --exclude="i01r01c01s[01-08]"

#SBATCH --mem=80G

source /etc/profile.d/modules.sh

module load slurm_setup

echo 'num_nodes:' $SLURM_JOB_NUM_NODES 'ntasks:' $SLURM_NTASKS
ulimit -Ss 2097152

export MP_SINGLE_THREAD=no
unset KMP_AFFINITY
export OMP_NUM_THREADS=48


script_dir=../rapid-earthquake-dynamics/
$script_dir/dynworkflow/compute_gof_fault_slip.py extracted_output/dyn_ extracted_output/dyn-kinmod_extracted-fault.xdmf
$script_dir/dynworkflow/compute_percentage_supershear.py extracted_output/dyn_ tmp/axitra_velocity_model.txt
$script_dir/dynworkflow/add_source_files_to_waveform_config.py
export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo)
$script_dir/submodules/seismic-waveform-factory/scripts/generate_figure_synthetics.py waveforms_config_sources.ini
$script_dir/dynworkflow/compile_scenario_macro_properties.py extracted_output


