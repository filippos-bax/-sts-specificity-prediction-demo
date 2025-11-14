from Bio import AlignIO
import pandas as pd
from src.io import load_config

def read_msa(file_path: str) -> pd.DataFrame:
    """
    Reads a multiple sequence alignment (MSA) file and returns a DataFrame.

    Args:
        file_path (str): The path to the MSA file.

    Returns:
        pd.DataFrame: A DataFrame containing the MSA data.
    """
    # Load configuration to handle problematic seq_ids
    cfg = load_config()
    problematic_seq_ids = cfg.get("problematic_seq_ids", {})

    try:
        alignment = AlignIO.read(file_path, "clustal")
        data = {
            "seq_id": [record.id for record in alignment],
            "alignment": [str(record.seq) for record in alignment]
        }
        df = pd.DataFrame(data)

        # Replace problematic seq_ids with their correct versions
        df['seq_id'] = df['seq_id'].replace(problematic_seq_ids)
        
        return df
    except Exception as e:
        print(f"Error reading MSA file: {e}")
        return pd.DataFrame(columns=['seq_id', 'alignment'])

