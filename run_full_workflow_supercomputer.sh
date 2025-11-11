#!/bin/bash

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich

echo parameters_fl33.par > fl33.txt
script_dir=../rapid-earthquake-dynamics/

# Function to wait for a job to finish
wait_for_job() {
    local job_id=$1
    while :; do
        # Check if the job is still running or pending
        job_status=$(squeue --job=$job_id --noheader)
        if [[ -z "$job_status" ]]; then
            echo "Job $job_id has finished."
            break
        fi
        echo "Waiting for job $job_id to finish..."
        sleep 10
    done
}

PARTITION=test
export order=5

hostname=$(hostname)

if [[ $hostname == uan* ]]; then
    supercomputer="LUMI"
elif [[ $hostname == login* ]]; then
    supercomputer="supermucNG1"
else
    echo "Error: Unknown hostname '$hostname'" >&2
    exit 1
fi

echo "Detected supercomputer: $supercomputer"



if [[ -n "$1" ]]; then
  # If argument $1 is given, use it as job ID
  job1_id="$1"
  echo "Using provided job ID: $job1_id"
else
  # Otherwise, run the pseudo-static step
  job1_id=$(sbatch "${script_dir}/scripts/${supercomputer}/job_test.sh" fl33.txt | awk '{print $NF}')
  echo "Submitted pseudo-static job with ID: $job1_id"
  wait_for_job "$job1_id"
fi
output=$(${script_dir}/dynworkflow/get_walltime_and_ranks_aggregated_job.py logs/${job1_id}.fl33.out)
walltime=$(echo "$output" | grep "Walltime:" | awk '{print $2}')
nodes=$(echo "$output" | grep "Chosen nodes:" | awk '{print $3}')

echo "Using walltime: $walltime"
echo "Using nodes: $nodes"

job2_id=$(sbatch --partition=$PARTITION ${script_dir}/scripts/${supercomputer}/create_parameters.sh | awk '{print $NF}')

PARTITION=micro

job3_id=$(sbatch --time=$walltime --nodes=$nodes --dependency=afterok:$job2_id ${script_dir}/scripts/${supercomputer}/aggregated_jobs.sh part_1.txt | awk '{print $NF}')
job4_id=$(sbatch --partition=$PARTITION --dependency=afterok:$job3_id ${script_dir}/scripts/${supercomputer}/compact_output_and_generate_PS.sh | awk '{print $NF}')
job5_id=$(sbatch --partition=$PARTITION --dependency=afterok:$job4_id ${script_dir}/scripts/${supercomputer}/generate_syn.sh | awk '{print $NF}')
