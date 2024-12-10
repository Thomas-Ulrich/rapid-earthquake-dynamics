#!/bin/bash
echo parameters_fl33.par > fl33.txt
script_dir=../rapid-earthquake-dynamics/
job1_id=$(sbatch ${script_dir}/scripts/job_NG_test.sh fl33.txt| awk '{print $NF}')
echo "Job 1 submitted with ID: $job1_id"
job2_id=$(sbatch --dependency=afterok:$job1_id ${script_dir}/scripts/job_NG_create_parameters.sh | awk '{print $NF}')
echo "Job 2 submitted with ID: $job2_id"
job3_id=$(sbatch --dependency=afterok:$job2_id ${script_dir}/scripts/job_NG.sh part_1.txt | awk '{print $NF}')
job4_id=$(sbatch --dependency=afterok:$job2_id ${script_dir}/scripts/job_NG.sh part_2.txt | awk '{print $NF}')
job5_id=$(sbatch --dependency=afterok:$job2_id ${script_dir}/scripts/job_NG.sh part_3.txt | awk '{print $NF}')

