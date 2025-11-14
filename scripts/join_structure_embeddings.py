import numpy as np
import pandas as pd
import ast

# Load the metadata DataFrame with sequence embeddings
meta_df = pd.read_pickle("data/processed/04-with-sequence-embeddings/metadata_with_embeddings.pkl")

# Load the structure embeddings DataFrame
str_embeddings_df = pd.read_csv("data/processed/06-structure-embeddings/structure_embeddings.csv")

# Ensure the 'seq_id' column is present in both DataFrames
if 'seq_id' not in meta_df.columns or 'seq_id' not in str_embeddings_df.columns:
    raise ValueError("Both DataFrames must contain a 'seq_id' column.")

# Make sure that the 'seq_id' values both dataframes are overlapping completely
overlap = set(meta_df['seq_id']).intersection(set(str_embeddings_df['seq_id']))
if len(overlap) != len(meta_df['seq_id']):
    raise ValueError("The 'seq_id' values in the two DataFrames do not overlap completely.")

# Turn the 'structure_embeddings' column in str_embeddings_df from string representation to list
str_embeddings_df['structure_embeddings'] = str_embeddings_df['structure_embeddings'].apply(ast.literal_eval)

# Pass the structure embeddings DataFrame to the metadata DataFrame
meta_df = meta_df.merge(str_embeddings_df, on='seq_id', how='left')

# Save the updated metadata DataFrame with structure embeddings
meta_df.to_pickle("data/processed/06-structure-embeddings/metadata_with_structure_embeddings.pkl")