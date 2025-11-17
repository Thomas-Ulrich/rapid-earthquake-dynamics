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

# Function to return Slurm dependency option if job ID exists
get_dependency() {
  local prev_job_id=$1
  if [[ -n "$prev_job_id" ]]; then
    echo "--dependency=afterok:$prev_job_id"
  else
    echo ""
  fi
}

# Function to get walltime, nodes, and nodes per simulation
get_resources() {
  local logfile=$1
  local max_hours=$2

  output=$(${script_dir}/src/dynworkflow/get_walltime_and_ranks_aggregated_job.py "$logfile" --max_hours "$max_hours")
  walltime=$(echo "$output" | grep "Walltime:" | awk '{print $2}')
  nodes=$(echo "$output" | grep "Chosen nodes:" | awk '{print $3}')
  nodes_per_sim=$(echo "$output" | grep "Nodes per sim:" | awk '{print $4}')

  echo "Using walltime: $walltime"
  echo "Using nodes: $nodes"
  echo "Using nodes per sim: $nodes_per_sim"
}

export order=5

# Default start step
start_from="job1"

echo "All arguments: $@"

# Parse optional arguments
for arg in "$@"; do
  case $arg in
    start_from=*)
      start_from="${arg#*=}"
      shift
      ;;
    job1_log_file=*)
      job1_log_file="${arg#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 start_from=<value> job1_log_file=<value>"
      echo "job1: pseudo-static step."
      echo "job2: DR input file generation."
      echo "job3: aggregated DR simulations."
      echo "job4: compact outputs and compute multi CMT solutions."
      echo "job5: compute GOF metrics and synthetic waveforms and rank models."
      exit 1
      ;;
  esac
done

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

echo "start_from = $start_from"
echo "job1_log_file = $job1_log_file"

# Step 1: Pseudo-static
if [[ "$start_from" == "job1" || "$start_from" == "all" ]]; then
  if [[ -n "$job1_log_file" ]]; then
    echo "Using provided job log file: $job1_log_file"
  else
    job1_id=$(sbatch "${script_dir}/scripts/${supercomputer}/job_test.sh" fl33.txt | awk '{print $NF}')
    echo "Submitted pseudo-static job with ID: $job1_id"
    wait_for_job "$job1_id"
  fi
fi

# Step 2: Get walltime/nodes and create parameters
if [[ "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
  if [[ -n "$job1_log_file" ]]; then
    get_resources "$job1_log_file" "$max_hours"
  else
    get_resources "logs/${job1_id}.fl33.out" "$max_hours"
  fi
  job2_id=$(sbatch --partition=$PARTITION1 ${script_dir}/scripts/${supercomputer}/create_parameters.sh | awk '{print $NF}')
else
  echo "Skipping job1/job2 setup."
fi

# Step 3: Aggregated job
if [[ "$start_from" == "job3" || "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then

  if [[ "$start_from" == "job3" ]]; then
    if [[ -n "$job1_log_file" ]]; then
      get_resources "$job1_log_file" "$max_hours"
    else
      echo "error job1_log_file ($job1_log_file) is not set or does not exist"
      exit 1
    fi
  fi

  dep=$(get_dependency "$job2_id")
  job3_id=$(sbatch --time=$walltime --nodes=$nodes $dep \
    ${script_dir}/scripts/${supercomputer}/aggregated_jobs.sh part_1.txt ${nodes_per_sim} | awk '{print $NF}')
  echo "Submitted job3: $job3_id"
fi

# Step 4: Compact output
if [[ "$start_from" == "job4" || "$start_from" == "job3" || "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
  dep=$(get_dependency "$job3_id")
  job4_id=$(sbatch --partition=$PARTITION2 $dep \
    ${script_dir}/scripts/${supercomputer}/compact_output_and_generate_PS.sh | awk '{print $NF}')
  echo "Submitted job4: $job4_id"
fi

# Step 5: Generate synthetic data
if [[ "$start_from" == "job5" || "$start_from" == "job4" || "$start_from" == "job3" || "$start_from" == "job2" || "$start_from" == "job1" || "$start_from" == "all" ]]; then
  dep=$(get_dependency "$job4_id")
  job5_id=$(sbatch --partition=$PARTITION2 $dep \
    ${script_dir}/scripts/${supercomputer}/generate_syn.sh | awk '{print $NF}')
  echo "Submitted job5: $job5_id"
fi
