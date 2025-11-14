import pandas as pd
import numpy as np
from src.read_msa import read_msa
from src.read_nmd import read_nmd


def merge_msa_nmd(msa_file: str, nmd_folder: str) -> pd.DataFrame:
    # Read MSA file
    msa_df = read_msa(msa_file)
    print(f"MSA DataFrame shape: {msa_df.shape}")
    
    # Read NMD file
    nmd_df = read_nmd(nmd_folder)
    print(f"NMD DataFrame shape: {nmd_df.shape}")
    
    # Merge MSA and NMD dataframes on 'seq_id'
    merged_df = pd.merge(msa_df, nmd_df, on='seq_id', how='left')
        
    return merged_df

def make_mode_embeddings(df):

    # Create mode embeddings based on the alignment and mode data
    df['mode_embeddings'] = None 

    for index, row in df.iterrows():
        alignment = row['alignment']
        mode = row['mode']
        resids = row['resids']
        
        i = 0
        j = 0
        embedding = []
        for aa in alignment:
            if aa == '-':
                aa = '0'
            # If letter is 
            elif aa.isalpha():
                i += 1
                if i < resids[0]:
                    aa = '0'
                elif i > resids[-1]:
                    aa = '0'
                else:
                    aa = mode[j]
                    j += 1
            embedding.append(aa)
            
        df.at[index, 'mode_embeddings'] = embedding

    return df
