#!/usr/bin/env python3
"""
Combine heterogeneous metadata files into one standardized table.
"""

import os, sys
from src.io import load_config, read_fasta_records
from src.metadata import read_dataset, resolve_duplicates
import pandas as pd
from Bio import SeqIO
import argparse




def main():
    p = argparse.ArgumentParser(
        description="Combine metadata from multiple datasets."
    )
    p.add_argument(
        "-c", "--config", default="config/config.yaml",
        help="Path to YAML config"
    )
    p.add_argument(
        "-o", "--output", default="data/processed/metadata.csv",
        help="Output combined metadata CSV"
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    dfs = []

    for ds_name, ds_cfg in cfg["datasets"].items():
            if ds_name != 'bacterial':
                print(f"→ Loading dataset '{ds_name}' from {ds_cfg['path']} …")
                df = read_dataset(ds_name, ds_cfg)
                missing = set(cfg["required_columns"]) - set(df.columns)
                if missing:
                    print(f"[ERROR] {ds_name} is missing columns: {missing}")
                    for col in missing:
                        df[col] = pd.NA
                if ds_name != 'mts':
                    df['seq_id'] = (
                            df['sequence']
                            .str.split("\n").str[0]
                            .str.lstrip(">")
                            .str.split().str[0]
                    )
                # MTS dataset
                elif ds_name == 'mts':
                    df['seq_id'] = (
                            df['seq_id']
                            .str.strip()
                    )
                    for seq_id in df['seq_id']:
                        combined_fasta_genrator = read_fasta_records(cfg["paths"]["final_processed_combined_dataset"])
                        for record in combined_fasta_genrator:
                            generator_id = record.split('\n')[0].lstrip('>').strip()
                            if seq_id == generator_id:
                                df.loc[df['seq_id'] == generator_id, 'sequence'] = record
                    df['length'] = 'MTS'
                    df['type'] = 'class I'
                    df['main_product'] = df.iloc[:, 5:28].idxmax(axis=1)
                # STS dataset
                if ds_name == 'sts':
                    df['length'] = 'STS'
                    df['type'] = 'class I'
                # DTS dataset
                if ds_name == 'dts':
                    df['length'] = 'DTS'
                # Poplar dataset
                if ds_name == 'poplar':
                    non_standard_columns = ds_cfg.get("non-standard_columns", {})
                    df = df.rename(columns=non_standard_columns)
                    df.loc[df['length'] == 'STS', 'main_product'] = df.loc[df['length'] == 'STS', 'fpp_product']
                    df.loc[df['length'] == 'MTS', 'main_product'] = df.loc[df['length'] == 'MTS', 'gpp_product']
                    df.loc[df['length'] == 'MTS/STS', 'main_product'] = df.loc[df['length'] == 'MTS/STS', 'gpp_product'] + df.loc[df['length'] == 'MTS/STS', 'fpp_product']
                keep = cfg["required_columns"] + ["dataset"]
                dfs.append(df[keep])
            elif ds_name == 'bacterial':
                for bc_name, bc_cfg in ds_cfg.items():
                    print(f"→ Loading dataset '{bc_name}' from {bc_cfg['path']} …")
                    df = read_dataset(bc_name, bc_cfg)
                    missing = set(cfg["required_columns"]) - set(df.columns)
                    if missing:
                        print(f"[ERROR] {bc_name} is missing columns: {missing}")
                        for col in missing:
                            df[col] = pd.NA
                    df['seq_id'] = (
                            df['sequence']
                            .str.split("\n").str[0]
                            .str.lstrip(">")
                            .str.split().str[0]
                    )
                    if bc_name == 'bacterial_1':
                        non_standard_columns = bc_cfg.get("non-standard_columns", {})
                        df = df.rename(columns=non_standard_columns)
                        df['taxon'] = df['taxon_1'] + ' ' + df['taxon_2']
                        df.loc[df['length'] == 'Sesqui', 'length'] = 'STS'
                        df.loc[df['length'] == 'Di', 'length'] = 'DTS'
                        df.loc[df['length'] == 'Mono', 'length'] = 'MTS'
                        df.loc[df['length'] == 'Mono ', 'length'] = 'MTS'
                        df.loc[df['length'] == 'Mono/Sesqui', 'length'] = 'MTS/STS'
                        df.loc[df['length'] == 'Sesquar', 'length'] = 'SESQUAR-TS'
                        df.loc[df['length'] == 'Sester', 'length'] = 'SESTER-TS'
                    df['dataset'] = 'bacterial'
                    keep = cfg["required_columns"] + ["dataset"]
                    dfs.append(df[keep])

    combined = pd.concat(dfs, ignore_index=True, sort=False)


    # Load the FASTA records once outside the loop
    combined_fasta_generator = list(read_fasta_records(cfg["paths"]["final_processed_combined_dataset"]))
    combined_fasta_generator = [''.join(record.split('\n')[1:]) for record in combined_fasta_generator]

    # Initialize counters
    counters = {
        'bacterial': 0,
        'sts': 0,
        'mts': 0,
        'dts': 0,
        'poplar': 0
    }

    indices_to_drop = []

    # Check whether each sequence exists in the combined fasta file
    for idx, row in combined.iterrows():
        seq = row['sequence']
        seq_id = row['seq_id']
        seq_line = ''.join(seq.split('\n')[1:])
        dataset = row['dataset']

        if seq_line not in combined_fasta_generator:
            rectified_seq = seq_line.replace('X', 'G')
            if rectified_seq not in combined_fasta_generator:
                if dataset in counters:
                    counters[dataset] += 1
                    print(f"Missing sequence in {dataset} dataset: {seq_id}")
                    if dataset == 'mts':
                        print(f'{seq_id}: {seq}')
                indices_to_drop.append(idx)
            else:
                print (idx)
                seq_sepatate_lines = seq.split('\n')
                seq_sepatate_lines[1:] = [line.replace('X', 'G') for line in seq_sepatate_lines[1:]]
                combined.at[idx, 'sequence'] = '\n'.join(seq_sepatate_lines)


    combined = combined.drop(index=indices_to_drop).reset_index(drop=True)

    for name, count in counters.items():
        print(f'{name}_counter: {count}')
    print (sum(counters.values()))

    print (combined['sequence'].duplicated().any())
    print (combined[combined['sequence'].duplicated(keep=False)])

    combined.loc[combined['dataset'] == 'sts', 'type'] = 'Type I'
    combined.loc[combined['dataset'] == 'mts', 'type'] = 'Type I'
    combined.loc[combined['dataset'] == 'poplar', 'type'] = 'Type I'
    combined.loc[(combined['type'] == 'class I diTPS') & (combined['dataset'] == 'dts'), 'type'] = 'Type I'
    combined.loc[(combined['type'] == 'class II diTPS') & (combined['dataset'] == 'dts'), 'type'] = 'Type II'
    combined.loc[(combined['type'] == 'bifunctional diTPS') & (combined['dataset'] == 'dts'), 'type'] = 'Type I/II'


    # Create an 'acyclic status' column filled with pd.NA, fill it with labels based on the 'acyclic' column
    df['acyclic_status'] = pd.NA

    mts_mask = combined['dataset'] == 'mts'
    sts_mask = combined['dataset'] == 'sts'

    acyclic_numeric = pd.to_numeric(combined.loc[mts_mask, 'acyclic'], errors='coerce')

    combined.loc[mts_mask & (acyclic_numeric > 0.5), 'acyclic_status'] = 'acyclic'
    combined.loc[mts_mask & (acyclic_numeric <= 0.5), 'acyclic_status'] = 'cyclic'


    combined.loc[sts_mask & (combined['acyclic'] == 'acyclic, 6,1-'), 'acyclic_status'] = 'acyclic/cyclic'
    combined.loc[sts_mask & (combined['acyclic'] == 'acyclic'), 'acyclic_status'] = 'acyclic'
    combined.loc[sts_mask & (combined['acyclic'] != 'acyclic'), 'acyclic_status'] = 'cyclic'


    combined = combined.groupby('sequence', group_keys=False).apply(resolve_duplicates).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"\n✅ Wrote combined metadata ({len(combined)} rows) to {args.output}")


if __name__ == "__main__":
    main()
