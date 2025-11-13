#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

echo parameters_fl33.par >fl33.txt
script_dir=../rapid-earthquake-dynamics/

# Function to wait for a job to finish
wait_for_job() {
    local job_id=$1
    while :; do
        job_status=$(squeue --job=$job_id --noheader)
        if [[ -z "$job_status" ]]; then
            echo "Job $job_id has finished."
            break
        fi
        echo "Waiting for job $job_id to finish..."
        sleep 10
    done
}

export order=5

hostname=$(hostname)

if [[ $hostname == uan* ]]; then
    supercomputer="LUMI-G"
    max_hours=1
    PARTITION1="small-g"
    PARTITION2="small-g"
elif [[ $hostname == login* ]]; then
    supercomputer="supermucNG1"
    max_hours=4
    PARTITION1=test
    PARTITION2=micro
else
    echo "Error: Unknown hostname '$hostname'" >&2
    exit 1
fi

echo "Detected supercomputer: $supercomputer"

# Default start step
start_from="job1"

# Parse optional arguments
for arg in "$@"; do
    case $arg in
        start_from=*)
            start_from="${arg#*=}"
            shift
            ;;
        job1_id=*)
            job1_id="${arg#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            ;;
    esac
done

echo "Starting workflow from: $start_from"

# Step 1: Pseudo-static
if [[ "$start_from" == "job1" || "$start_from" == "all" ]]; then
    if [[ -n "$job1_id" ]]; then
        echo "Using provided job ID: $job1_id"
    else
        job1_id=$(sbatch "${script_dir}/scripts/${supercomputer}/job_test.sh" fl33.txt | awk '{print $NF}')
        echo "Submitted pseudo-static job with ID: $job1_id"
        wait_for_job "$job1_id"
    fi
fi

# Step 2: Get walltime/nodes and create parameters
if [[ "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
    output=$(${script_dir}/dynworkflow/get_walltime_and_ranks_aggregated_job.py logs/${job1_id}.fl33.out --max_hours $max_hours)
    walltime=$(echo "$output" | grep "Walltime:" | awk '{print $2}')
    nodes=$(echo "$output" | grep "Chosen nodes:" | awk '{print $3}')

    echo "Using walltime: $walltime"
    echo "Using nodes: $nodes"

    job2_id=$(sbatch --partition=$PARTITION1 ${script_dir}/scripts/${supercomputer}/create_parameters.sh | awk '{print $NF}')
else
    echo "Skipping job1/job2 setup."
fi

# Step 3: Aggregated job
if [[ "$start_from" == "job3" || "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
    job3_id=$(sbatch --time=$walltime --nodes=$nodes --dependency=afterok:${job2_id:-none} \
        ${script_dir}/scripts/${supercomputer}/aggregated_jobs.sh part_1.txt | awk '{print $NF}')
    echo "Submitted job3: $job3_id"
fi

# Step 4: Compact output
if [[ "$start_from" == "job4" || "$start_from" == "job3" || "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
    job4_id=$(sbatch --partition=$PARTITION2 --dependency=afterok:${job3_id:-none} \
        ${script_dir}/scripts/${supercomputer}/compact_output_and_generate_PS.sh | awk '{print $NF}')
    echo "Submitted job4: $job4_id"
fi

# Step 5: Generate synthetic data
if [[ "$start_from" == "job5" || "$start_from" == "job4" || "$start_from" == "job3" || "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
    job5_id=$(sbatch --partition=$PARTITION2 --dependency=afterok:${job4_id:-none} \
        ${script_dir}/scripts/${supercomputer}/generate_syn.sh | awk '{print $NF}')
    echo "Submitted job5: $job5_id"
fi
