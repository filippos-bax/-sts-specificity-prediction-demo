# src/metadata.py

import pandas as pd
import sys
import yaml
from src.io import load_config

def read_dataset(ds_name, ds_cfg):
    """
    Read one dataset according to its config, rename columns to standard keys,
    add a `dataset` column, and return a pandas DataFrame.
    """
    path = ds_cfg["path"]
    ftype = ds_cfg.get("file_type", "csv").lower()
    # 1) Load
    if ftype == "excel":
        df = pd.read_excel(path, sheet_name=ds_cfg.get("sheet_name", 0))
    elif ftype == "tsv":
        df = pd.read_csv(path, sep="\t")
    else:  # assume CSV
        df = pd.read_csv(path)

    # 2) Rename columns
    #    config columns: raw_name → standard_name
    raw2std = ds_cfg.get("columns", {})
    # keep only the columns we know how to rename
    df = df.rename(columns=raw2std)

    # 3) Check required columns exist
    #    (they’ll be enforced in main)

    # 4) Tag with dataset name
    df["dataset"] = ds_name

    return df

def resolve_duplicates(group):
    if len(group) == 1:
        return group  # No duplicates
    
    unique_values = group['main_product'].unique()
    
    # If all values are the same, keep just the first
    if len(unique_values) == 1:
        return group.iloc[[0]]
    
    # If values differ, join them and keep one row
    joined_value = ','.join(map(str, sorted(set(unique_values))))
    new_row = group.iloc[0].copy()
    new_row['main_product'] = joined_value
    return pd.DataFrame([new_row])
