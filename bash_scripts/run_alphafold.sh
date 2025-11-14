#!/bin/bash

#SBATCH --job-name=alphafold_multi_node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100
#SBATCH --output=/home/fbaxevanis/slurm_logs/alphafold/alphafold_multi_node_%A.out
#SBATCH --error=/home/fbaxevanis/slurm_logs/alphafold/alphafold_multi_node_%A.err
#SBATCH --mail-user=f.baxevanis@student.vu.nl
#SBATCH --mail-type=ALL

# Load modules and ensure the environment is propagated
module purge
module load "2022" 
module load "AlphaFold/2.3.1-foss-2022a-CUDA-11.7.0"

# Define variables
data_root="/projects/2/managed_datasets/AlphaFold/2.3.1"
output_dir="/scratch-shared/$USER/alphafold_outputs"
input_dir="/scratch-shared/$USER/lost_seqs"

# Copy input files to scratch
rsync -av --progress "data/processed/02-no-abnormal-aminoacids/" "$input_dir/"

# Create output directory
mkdir -p "$output_dir"
find "$input_dir" -name "*.fasta" > file_list.txt
total_files=$(wc -l < file_list.txt)

# Pre-create task-specific directories and copy parameters
for idx in $(seq 0 $((SLURM_NTASKS-1))); do
    task_scratch="/scratch-shared/$USER/task_${idx}"
    mkdir -p "$task_scratch"
    cp -r "$data_root/params" "$task_scratch/"
done

# Export variables so they are available inside srun
export output_dir
export total_files

# Distribute files dynamically among tasks
srun bash -c '
    idx=$((SLURM_PROCID))
    num_tasks=$((SLURM_NTASKS))
    task_scratch="/scratch-shared/$USER/task_${idx}"
    export ALPHAFOLD_PARAMS_DIR="$task_scratch/params"
    gpu_id=$((idx % 4))
    export CUDA_VISIBLE_DEVICES=$gpu_id

    # Assign specific files to this task
    for file_index in $(seq $((idx + 1)) $num_tasks $total_files); do
        input_file=$(sed -n "${file_index}p" file_list.txt)
        echo "Task $idx using GPU $CUDA_VISIBLE_DEVICES for $input_file"

        alphafold \
            --fasta_paths "$input_file" \
            --output_dir "$output_dir" \
            --db_preset full_dbs \
            --data_dir "$task_scratch" \
            --max_template_date 2023-03-20
    done
'

# Copy output files back to home
rsync -av --include='*/' --include='relaxed_model*.pdb' --exclude='msas/' --exclude='*' /scratch-shared/$USER/alphafold_outputs/ data/processed/05-structures/
