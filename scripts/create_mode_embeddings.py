import pandas as pd
from src.make_mode_embeddings import merge_msa_nmd, make_mode_embeddings
from src.io import load_config
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create mode embeddings for the entire dataset.")
    parser.add_argument('--meta', type=str, required=True, help='Path to the metadata file')
    return parser.parse_args()

def main():
    args = parse_args()
    meta_df = pd.read_pickle(args.meta)
    config = load_config()
    paths = config.get('paths', {})
    msa_folder = paths['msa_folder']
    msa_file = f'{msa_folder}/all.aln-clustal_num'
    nmd_folder = paths['nmd_folder']
    mode_df = merge_msa_nmd(msa_file, nmd_folder)
    mode_df = make_mode_embeddings(mode_df)
    meta_df = pd.merge(meta_df, mode_df, on='seq_id', how='left')
    meta_df.to_pickle("data/processed/12-mode-embeddings/metadata_with_mode_embeddings.pkl")

if __name__ == "__main__":
    main()