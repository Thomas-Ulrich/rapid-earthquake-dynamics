#!/usr/bin/env bash
#
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024–2025 Thomas Ulrich
#
# Aggregated SeisSol job launcher

set -euo pipefail

##############################
# Job splitting logic
##############################

nodes_per_job=${2:-1}
ndivide=$((SLURM_JOB_NUM_NODES / ${nodes_per_job}))
tasks_per_job=$((nodes_per_job * 8))

echo "Total nodes: $SLURM_JOB_NUM_NODES"
echo "Dividing into $ndivide parallel jobs, each using $nodes_per_job nodes ($tasks_per_job tasks)."
mkdir logs/${SLURM_JOB_ID}_runs
##############################
# Collect parameter files
##############################

if [[ -n "${1:-}" ]]; then
  part_file=$1
  echo "$(date '+%Y-%m-%d %H:%M:%S') - Reading parameter files from: $part_file"
  mapfile -t files <"$part_file"
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
for ((i = 0; i < ndivide; i++)); do
  start=$((i * nodes_per_job))
  subset_nodes=("${all_nodes[@]:$start:$nodes_per_job}")
  node_subsets+=("$(
    IFS=,
    echo "${subset_nodes[*]}"
  )")
done

echo "node subsets ${node_subsets[*]}"

declare -A active_jobs=()      # pid → node_subset mapping
declare -A parameter_lookup=() # pid → parameter_name
declare -a failed_jobs=()      # store failed parameter files

##############################
# Function to run one SeisSol job
##############################

run_file() {
  local filename=$1
  local local_counter=$2
  local node_subset=$3

  local counter0
  counter0=$(printf "%05d" "$local_counter")
  local id
  id=$(echo "$filename" | sed -n 's/^parameters_dyn_\([0-9]\{4\}\)_.*\.par/\1/p')

  echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching $filename on nodes: $node_subset"
  logfile="${SLURM_JOB_ID}.$counter0.$id.out"
  srun_cmd &
  pid=$!

  # avoid “zombie” entry in active_jobs
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Warning: srun for $filename failed to start properly."
    unset active_jobs["$pid"]
    failed_jobs+=("$fn")
  else
    active_jobs["$pid"]="$node_subset"
    parameter_lookup["$pid"]="$filename"
  fi
}

check_active_jobs() {
  for pid in "${!active_jobs[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"
      exit_code=$?
      fn=${parameter_lookup[$pid]}
      echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished $fn on nodes: ${active_jobs[$pid]}"
      # track failed jobs
      if [[ $exit_code -ne 0 ]]; then
        failed_jobs+=("$fn")
      else
        # remove fn from failed_jobs if it exists
        tmp=()
        for f in "${failed_jobs[@]}"; do
          [[ "$f" != "$fn" ]] && tmp+=("$f")
        done
        failed_jobs=("${tmp[@]}")
      fi
      unset active_jobs[$pid]
      unset parameter_lookup[$pid]
      return 0 # indicate a job finished
    fi
  done
  return 1 # no jobs finished
}

##############################
# Job scheduling loop
##############################
run_jobs() {
  local files=("$@") # all arguments become the array "files"
  for filename in "${files[@]}"; do
    # Wait if all subsets busy
    while ((${#active_jobs[@]} >= ndivide)); do
      # Wait for any job to finish
      while true; do
        check_active_jobs && break # break the outer loop if a job finished
        sleep 1                    # avoid busy-waiting
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
}

wait_for_remaining_jobs() {
  # Wait for remaining background jobs and print finished jobs in the order they finish
  last_print_time=$(date +%s)

  # Wait for remaining background jobs and print finished jobs in the order they finish
  while ((${#active_jobs[@]} > 0)); do
    while check_active_jobs; do
      : # loop to handle multiple finished jobs that may have accumulated
    done
    # Print debug info every 60 seconds
    now=$(date +%s)
    if ((now - last_print_time >= 60)); then
      echo "----- STATUS $(date '+%Y-%m-%d %H:%M:%S') -----"
      echo "Active jobs: ${#active_jobs[@]}"
      echo "Tracked parameters: ${#parameter_lookup[@]}"
      echo "PIDs still running: ${!active_jobs[@]}"
      for pid in "${!active_jobs[@]}"; do
        echo "  PID $pid → nodes: ${active_jobs[$pid]}, file: ${parameter_lookup[$pid]}"
      done
      echo "---------------------------------------------"
      last_print_time=$now
    fi
    sleep 1 # avoid busy-waiting
  done
}

counter=0
run_jobs "${files[@]}"
wait_for_remaining_jobs

if [[ ${#failed_jobs[@]} -gt 0 ]]; then
  echo "---------------------------------------------"
  echo "${#failed_jobs[@]} job(s) failed:"
  for f in "${failed_jobs[@]}"; do
    echo " - $f"
  done
  echo "---------------------------------------------"
else
  echo "All jobs completed successfully."
fi

# Wait for remaining background jobs
echo "$(date '+%Y-%m-%d %H:%M:%S') - All parameter files processed."
