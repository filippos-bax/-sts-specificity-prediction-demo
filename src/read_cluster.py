import pandas as pd

def read_cluster(file_path: str) -> pd.DataFrame:
    """
    Reads a cluster file in .tsv format produced by 
    mmseqs easy-cluster and returns a DataFrame.

    Args:
        file_path (str): The path to the cluster .tsv file.

    Returns:
        pd.DataFrame: A DataFrame containing the cluster data.
    """
    try:
        df = pd.read_table(file_path, sep='\t', header=None)
        df.columns = ['Cluster', 'Gene']
        return list(df['Cluster'].unique())
    except Exception as e:
        print(f"Error reading cluster file: {e}")
    

    