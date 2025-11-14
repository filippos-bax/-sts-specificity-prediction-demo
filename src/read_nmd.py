import os
import pandas as pd
import numpy as np

# Function that parsers the NMD files in a FOLDER and returns a DataFrame
def read_nmd(file_path: str) -> pd.DataFrame:
    """
    Reads a non-metric dimensionality reduction (NMD) file and returns a DataFrame.

    Args:
        file_path (str): The path to the NMD FOLDER.

    Returns:
        pd.DataFrame: A DataFrame containing the NMD data.
    """
    try:
        rows = []
        for file in os.listdir(file_path):
            if file.endswith('.nmd'):
                seq_id = os.path.splitext(file)[0].split(']')[1]
                seq_id = seq_id.replace('-', '|')
                nmd_file_path = os.path.join(file_path, file)
                with open(nmd_file_path, 'r') as f:
                    lines = f.readlines()
                    modes = []
                    resids = []
                    for line in lines:
                        if line.startswith("coordinates"):
                            coordinates = [float(coord) for coord in line.strip().split()[1:]]
                        elif line.startswith("mode"):
                            displacements = [float(coord) for coord in line.strip().split()[3:]]
                            mode = []
                            # Calculate the squared sum of displacements for each mode
                            for i in range(0, len(displacements), 3):
                                summ = (1000*displacements[i])**2 + \
                                    (1000*displacements[i+1])**2 + \
                                    (1000*displacements[i+2])**2
                                mode.append(summ)
                            modes.append(mode)
                        elif line.startswith("resids"):
                            resids = [float(resid) for resid in line.strip().split()[1:]]
                    array = np.vstack(modes)
                    array = np.round(array, 2)
                    array = array.mean(axis=0).tolist()
                rows.append({'seq_id': seq_id, 'mode': array, 'resids': resids, 'coordinates': coordinates})
        df = pd.DataFrame(rows)


        return df
    

    except Exception as e:
        print(f"Error reading NMD file: {e}")