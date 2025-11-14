#!/bin/bash

#SBATCH --job-name=gen_conformations
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=24
#SBATCH --time=4:00:00
#SBATCH --partition=genoa
#SBATCH --output=/home/fbaxevanis/slurm_logs/prody_nma/final_prody_%A.out
#SBATCH --error=/home/fbaxevanis/slurm_logs/prody_nma/final_prody_%A.err
#SBATCH --mail-user=f.baxevanis@student.vu.nl
#SBATCH --mail-type=ALL

# Load modules and ensure the environment is propagated
module purge
module load "2024"
module load Python/3.12.3-GCCcore-13.3.0

# Copy Python script to scratch
rsync -av --progress "scripts/sample_conformations.py" "/scratch-shared/$USER/"

# Define variables (directories already created)
structures_root="data/processed/05-structures"
python_script="/scratch-shared/fbaxevanis/sample_conformations.py"
input_dir="/scratch-shared/$USER/input_structures"
output_structures_dir="/scratch-shared/$USER/output_structures"
output_dssp_dir="/scratch-shared/$USER/output_dssp"
output_nmd_dir="/scratch-shared/$USER/output_nmd"
output_png_dir="/scratch-shared/$USER/output_png"
local_dssp_dir="data/processed/07-structure-related/dssp"
local_nmd_dir="data/processed/07-structure-related/nmd"
local_structures_dir="data/processed/08-conformations"
local_png_dir="data/processed/07-structure-related/png"

# Copy input files to scratch
rsync -av --progress "$structures_root/" "$input_dir/"

# Run mkdssp on input structures, save output to scratch
for file in "$input_dir"/*.pdb; do
    filename=$(basename "$file")
    output_file="$output_dssp_dir/${filename%.pdb}.dssp"
    echo "Processing $file to $output_file"
    mkdssp -i "$file" -o "$output_file"
done

# Purge modules and load the 2022 stack
module purge
module load "2022"
module load Python/3.10.4-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a

# Run the Python script on the input structures
for file in "$input_dir"/*.pdb; do
    filename=$(basename "$file")
    dssp_file="$output_dssp_dir/${filename%.pdb}.dssp"
    echo "Processing $file to $output_file"
    srun --ntasks=1 \
         --nodes=1 \
         --cpus-per-task=24 \
         --exclusive \
         python "$python_script" \
            --pdb "$file" \
            --pdb_outdir "$output_structures_dir" \
            --dssp "$dssp_file" \
            --nmd_outdir "$output_nmd_dir" \
            --png_outdir "$output_png_dir" &
done

# Waot for all tasks to complete
wait

# Copy results back to local directories
rsync -av --progress "$output_dssp_dir/" "$local_dssp_dir/"
rsync -av --progress "$output_nmd_dir/" "$local_nmd_dir/"
rsync -av --progress "$output_structures_dir/" "$local_structures_dir/"
rsync -av --progress "$output_png_dir/" "$local_png_dir/"