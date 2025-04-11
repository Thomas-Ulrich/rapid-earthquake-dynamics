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

get_scaled_walltime() {
  local job1_id=$1
  local scale_factor=${2:-120}
  local job_ranks=${3:-80}

  # Extract the kernel time in seconds from job output
  local kernel_time=$(grep "Total time spent in compute kernels" ${job1_id}*.out | awk '{print $(NF-10)}')

  if [[ -z "$kernel_time" ]]; then
    echo "Error: Could not extract kernel time from ${job1_id}*.out" >&2
    return 1
  fi

  # Compute scaled target time in seconds
  local target_time=$(echo "$kernel_time * $scale_factor / $job_ranks" | bc)

  # Format to HH:MM:SS
  printf -v walltime "%02d:%02d:%02d" \
    $(echo "$target_time/3600" | bc) \
    $(echo "($target_time%3600)/60" | bc) \
    $(echo "$target_time%60" | bc)

  echo "$walltime"
}

wait_for_job $job1_id

walltime=$(get_scaled_walltime "$job1_id" 120 80)
echo "Estimated walltime: $walltime"

job2_id=$(sbatch ${script_dir}/scripts/job_NG_create_parameters.sh | awk '{print $NF}')
job3_id=$(sbatch --time=$walltime --dependency=afterok:$job2_id ${script_dir}/scripts/aggregated_jobs_NG.sh part_1.txt | awk '{print $NF}')
job4_id=$(sbatch --dependency=afterok:$job3_id ${script_dir}/scripts/compact_output_and_generate_PS_NG.sh | awk '{print $NF}')
job5_id=$(sbatch --dependency=afterok:$job4_id ${script_dir}/scripts/generate_syn.sh | awk '{print $NF}')
