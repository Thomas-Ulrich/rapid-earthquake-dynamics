#!/bin/bash
echo parameters_fl33.par > fl33.txt
script_dir=../rapid-earthquake-dynamics/
job1_id=$(sbatch ${script_dir}/scripts/job_NG_test.sh fl33.txt| awk '{print $NF}')

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

get_scaled_walltime_and_ranks() {
  local job1_id=$1
  local scale_factor=${2:-120}
  local max_hours=${3:-2}

  local kernel_time=$(grep "Total time spent in compute kernels" ${job1_id}*.out | awk '{print $(12)}')
  if [[ -z "$kernel_time" ]]; then
    echo "Error: Could not extract kernel time from ${job1_id}*.out" >&2
    return 1
  fi

  local candidates=(20 40 80 160)
  local chosen_ranks=160
  local walltime=""

  for ranks in "${candidates[@]}"; do
    # 2.0 is a safety factor
    local target_time=$(echo "2.0 * $kernel_time * $scale_factor / ( 2 * $ranks )" | bc)

    local hours=$(echo "$target_time/3600" | bc)
    local minutes=$(echo "($target_time%3600)/60" | bc)
    local seconds=$(echo "$target_time%60" | bc)
    printf -v current_walltime "%02d:%02d:%02d" $hours $minutes $seconds

    if (( hours <= max_hours )); then
      chosen_ranks=$ranks
      walltime=$current_walltime
      break
    fi
  done

  # If none fit, use 160 and recalculate walltime
  if [[ -z "$walltime" ]]; then
    local target_time=$(echo "$kernel_time * $scale_factor / (2 * $chosen_ranks )" | bc)
    local hours=$(echo "$target_time/3600" | bc)
    local minutes=$(echo "($target_time%3600)/60" | bc)
    local seconds=$(echo "$target_time%60" | bc)
    printf -v walltime "%02d:%02d:%02d" $hours $minutes $seconds
  fi

  echo "$walltime $chosen_ranks"
}

wait_for_job $job1_id

read walltime ranks <<< $(get_scaled_walltime_and_ranks "$job1_id" 120 1)
echo "Using walltime: $walltime"
echo "Using ranks: $ranks"

job2_id=$(sbatch ${script_dir}/scripts/job_NG_create_parameters.sh | awk '{print $NF}')
job3_id=$(sbatch --time=$walltime --nodes=$ranks --dependency=afterok:$job2_id ${script_dir}/scripts/aggregated_jobs_NG.sh part_1.txt | awk '{print $NF}')
job4_id=$(sbatch --dependency=afterok:$job3_id ${script_dir}/scripts/compact_output_and_generate_PS_NG.sh | awk '{print $NF}')
job5_id=$(sbatch --dependency=afterok:$job4_id ${script_dir}/scripts/generate_syn.sh | awk '{print $NF}')
