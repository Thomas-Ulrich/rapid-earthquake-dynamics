#!/usr/bin/env bash
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich
#
# Aggregated SeisSol job launcher for LUMI-G

#SBATCH -J aggregated
#SBATCH --account=project_465002391
#SBATCH --time=00:30:00
#SBATCH -o logs/%j.%x.out
#SBATCH -e logs/%j.%x.out
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mail-type=all
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --export=ALL

set -euo pipefail

##############################
# Environment setup
##############################

if [[ ! -f ./select_gpu ]]; then

cat << EOF > select_gpu
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec "\$@"
EOF
chmod +x ./select_gpu

fi


CPU_BIND="7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1
export OMP_NUM_THREADS=3
export OMP_PLACES="cores(3)"
export OMP_PROC_BIND=close
export DEVICE_STACK_MEM_SIZE=4
export SEISSOL_FREE_CPUS_MASK="52-54,60-62,20-22,28-30,4-6,12-14,36-38,44-46"
export PATH=/project/project_465002391/ulrich/seissol_base/seissol/build:$PATH
ulimit -Ss 2097152
unset KMP_AFFINITY

ORDER=${order:-4}

##############################
# Job splitting logic
##############################

# Must use a number of nodes multiple of 2
if (( SLURM_JOB_NUM_NODES % 2 != 0 )); then
    echo "Error: $SLURM_JOB_NUM_NODES not a multiple of 2"
    exit 1
fi

if [ "$SLURM_JOB_NUM_NODES" -lt 60 ]; then
    ndivide=$(( SLURM_JOB_NUM_NODES / 1 ))
else
    ndivide=$(( SLURM_JOB_NUM_NODES / 2 ))
fi

nodes_per_job=$(( SLURM_JOB_NUM_NODES / ndivide ))
tasks_per_job=$(( nodes_per_job * 8 ))

echo "Total nodes: $SLURM_JOB_NUM_NODES"
echo "Dividing into $ndivide parallel jobs, each using $nodes_per_job nodes ($tasks_per_job tasks)."

##############################
# Collect parameter files
##############################

if [[ -n "${1:-}" ]]; then
    part_file=$1
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Reading parameter files from: $part_file"
    mapfile -t files < "$part_file"
else
    files=(parameters_dyn_*.par)
fi

num_files=${#files[@]}
echo "Found $num_files parameter files to process."

##############################
# Build node subsets
##############################

mapfile -t all_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    
echo "all nodes ${all_nodes[*]}"

node_subsets=()
for ((i=0; i<ndivide; i++)); do
    start=$((i * nodes_per_job))
    subset_nodes=("${all_nodes[@]:$start:$nodes_per_job}")
    node_subsets+=("$(IFS=,; echo "${subset_nodes[*]}")")
done

echo "node subsets ${node_subsets[*]}"

declare -A active_jobs=()  # pid → node_subset mapping
declare -A parameter_lookup=()  # pid → parameter_name

##############################
# Function to run one SeisSol job
##############################

run_file() {
    local filename=$1
    local counter=$2
    local node_subset=$3

    local counter0
    counter0=$(printf "%05d" "$counter")
    local id
    id=$(echo "$filename" | sed -n 's/^parameters_dyn_\([0-9]\{4\}\)_.*\.par/\1/p')

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching $filename on nodes: $node_subset"

    srun --nodes=$nodes_per_job --nodelist="$node_subset" \
         --ntasks=$tasks_per_job --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7\
         -o ./logs/$SLURM_JOB_ID.$counter0.$id.out --exclusive \
         --cpu-bind=mask_cpu:${CPU_BIND} \
         ./select_gpu SeisSol_Release_sgfx90a_hip_${ORDER}_elastic "$filename" &
    pid=$!
    active_jobs[$pid]="$node_subset"
    parameter_lookup[$pid]="$filename"
}

##############################
# Job scheduling loop
##############################

counter=0
for filename in "${files[@]}"; do
    # Wait if all subsets busy
    while (( ${#active_jobs[@]} >= ndivide )); do
# Wait for any job to finish
while true; do
    for pid in "${!active_jobs[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fn=${parameter_lookup[$pid]}
       	    echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished $fn on nodes: ${active_jobs[$pid]}"
            unset active_jobs[$pid]
            unset parameter_lookup[$pid]
            break 2  # exit the while+for loop, freeing one slot
        fi
    done
    sleep 1  # avoid busy-waiting
done
    done

    # Pick first free node subset
    for subset in "${node_subsets[@]}"; do
        if ! printf '%s\n' "${active_jobs[@]}" | grep -qx "$subset"; then
            run_file "$filename" "$counter" "$subset"
	    let counter=counter+1
            break
        fi
    done
done

# Wait for remaining background jobs and print finished jobs in the order they finish
while (( ${#active_jobs[@]} > 0 )); do
    for pid in "${!active_jobs[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            fn=${parameter_lookup[$pid]}
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished $fn on nodes: ${active_jobs[$pid]}"
            unset active_jobs[$pid]
            unset parameter_lookup[$pid]
            # break  # optional, can continue to check other jobs immediately
        fi
    done
    sleep 1  # avoid busy-waiting
done

# Wait for remaining background jobs
echo "$(date '+%Y-%m-%d %H:%M:%S') - All parameter files processed."
