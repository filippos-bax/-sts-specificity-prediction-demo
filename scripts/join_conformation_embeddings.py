import numpy as np
import pandas as pd
import os
import ast
from src.pooling_methods import max_pool_embeddings, mean_pool_embeddings
import argparse

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Join structure embeddings with metadata.')
    
    # Required arguments
    parser.add_argument('--input_meta', type=str, required=True, help='Path to the metadata DataFrame with sequence embeddings')
    parser.add_argument('--input_embeddings', type=str, required=True, help='Path to the structure embeddings DataFrame')
    
    # Optional arguments
    parser.add_argument('--pool', type=str, choices=['max', 'mean', 'original'], help='Pooling method for embeddings (default: max)')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    input_meta = args.input_meta
    input_embeddings = args.input_embeddings
    pooling_method = args.pool

    # Load the metadata DataFrame with sequence embeddings
    meta_df = pd.read_pickle(input_meta)

    # Load the structure embeddings DataFrame
    str_embeddings_df = pd.read_csv(input_embeddings)

    # Split the name of the conformation embeddings so that all conformations
    # of the same protein have the same seq_id
    str_embeddings_df[['seq_id', 'version']] = str_embeddings_df['seq_id'].str.split('*', expand=True)

    # Ensure the 'seq_id' column is present in both DataFrames
    if 'seq_id' not in meta_df.columns or 'seq_id' not in str_embeddings_df.columns:
        raise ValueError("Both DataFrames must contain a 'seq_id' column.")

    # Make sure that the 'seq_id' values both dataframes are overlapping completely
    overlap = set(meta_df['seq_id']).intersection(set(str_embeddings_df['seq_id']))
    if len(overlap) != len(meta_df['seq_id']):
        raise ValueError("The 'seq_id' values in the two DataFrames do not overlap completely.")
    
    # Drop old 'structure_embeddings' column
    meta_df_filtered = meta_df.drop(columns=['structure_embeddings'])

    # Turn the 'structure_embeddings' column in str_embeddings_df from string representation to list
    str_embeddings_df['structure_embeddings'] = str_embeddings_df['structure_embeddings'].apply(ast.literal_eval)

    # Pass the structure embeddings DataFrame to the metadata DataFrame
    merged_df = str_embeddings_df.merge(meta_df_filtered, on='seq_id', how='left')


    if pooling_method == 'max':
        # Apply max pooling to the structure embeddings
        pooled_df = merged_df.groupby('seq_id')['structure_embeddings'].apply(max_pool_embeddings).reset_index()
        merged_df = meta_df_filtered.merge(pooled_df, on='seq_id', how='left')
    elif pooling_method == 'mean':
        # Apply mean pooling to the structure embeddings
        pooled_df = merged_df.groupby('seq_id')['structure_embeddings'].apply(mean_pool_embeddings).reset_index()
        merged_df = meta_df_filtered.merge(pooled_df, on='seq_id', how='left')
    elif pooling_method == 'original':
        # Keep the original structure embeddings without pooling
        merged_df = merged_df[merged_df["version"] == 'v0'].drop(columns=['version'])

    # Save the updated metadata DataFrame with structure embeddings
    pool_ext = f'_{pooling_method}' if pooling_method else ''
    merged_df.to_pickle(f'data/processed/09-conformation-embeddings{pool_ext}.pkl')

if __name__ == "__main__":
    main()