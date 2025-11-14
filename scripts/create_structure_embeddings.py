import geometricus as gm
import numpy as np
import pandas as pd
import os
import time
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Create structure embeddings using Geometricus.')
    
    # Required arguments
    parser.add_argument('--i', type=str, dest='structure_folder', required=True, help='Path to the folder containing structure files')
    parser.add_argument('--o', type=str, dest='output_folder', required=True, help='Path to the output folder for saving embeddings')

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    structure_folder = args.structure_folder
    output_folder = args.output_folder
    structures = os.listdir(structure_folder)

    # Set parameters
    resolution = 1
    n_threads = 24
    kmer = 8
    radius = 5
    split = [
        gm.SplitInfo(gm.SplitType.KMER, kmer), 
        gm.SplitInfo(gm.SplitType.RADIUS, radius)
        ]

    # Start the timer
    start_time = time.time()

    # Run Geometricus
    invariants, _ = gm.get_invariants_for_structures(
        structure_folder, 
        n_threads = n_threads,
        split_infos = split,
        moment_types = ["O_3", "O_4", "O_5", "F"]
        )

    shapemer_class = gm.Geometricus.from_invariants(
        invariants, 
        protein_keys = structures, 
        resolution = resolution
        )

    # Get the count matrix
    shapemer_count_matrix = shapemer_class.get_count_matrix()

    # End the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


    # Remove the identifier from the structure names
    shapemer_class.protein_keys = [structure.split(']')[1] for structure in shapemer_class.protein_keys]

    # Remove the file extension from the structure names
    shapemer_class.protein_keys = [os.path.splitext(structure)[0] for structure in shapemer_class.protein_keys]

    # Replace the '-' character with '|' in the structure names
    shapemer_class.protein_keys = [structure.replace('-', '|') for structure in shapemer_class.protein_keys]

    # Normalize the count matrix for protein length
    shapemer_sum_matrix = np.sum(shapemer_count_matrix, axis=1, keepdims=True)
    shapemer_normalized_matrix = shapemer_count_matrix / shapemer_sum_matrix

    # Create a DataFrame to save the structure embeddings
    df = pd.DataFrame(shapemer_class.protein_keys, columns=['seq_id'])
    df['structure_embeddings'] = shapemer_normalized_matrix.tolist()

    # Make sure the saving directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the DataFrame to a .csv file
    df.to_csv(f'{output_folder}/structure_embeddings.csv', index=False)

    # Save the raw shapemer count matrix to a .npy file (optional)
    np.save(f'{output_folder}/shapemer_count_matrix.npy', shapemer_normalized_matrix)

if __name__ == "__main__":
    main()