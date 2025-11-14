import argparse
import os
from Bio import SeqIO
from src.io import load_config

"""
This script isolates datasets from a combined FASTA file based on specified dataset names,
then outputs the isolated datasets into a new combined FASTA file.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Isolate datasets from a combined FASTA file.")
    parser.add_argument("--input_file", help="Path to the input combined FASTA file.")
    parser.add_argument("--output_dir", help="Directory to save isolated datasets.")
    parser.add_argument("--datasets", nargs='+', required=True, 
                        help="List of datasets to isolate (e.g., bacterial, dts, sts).")
    return parser.parse_args()

def isolate_datasets(input_file, output_dir, datasets):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize a dictionary to hold SeqRecords for each dataset
    dataset_records = []

    # Load the configuration file to get dataset names if needed
    config = load_config()
    duplicate_sequences = config.get('duplicate_sequences', [])

    # Read the combined FASTA file and distribute records to corresponding datasets
    for record in SeqIO.parse(input_file, "fasta"):
        for dataset in datasets:
            if dataset in record.description:
                # Check if the record is a duplicate    
                if 'sts' in datasets and 'mts' in datasets and record.id in duplicate_sequences:
                    continue
                else:
                    dataset_records.append(record)
                break  # Stop checking once the record is assigned

    # Write the selected datasets to a new combined FASTA file
    output_name = '-'.join(datasets) + '.fasta'
    output_file = os.path.join(output_dir, output_name)
    with open(output_file, "w") as out_handle:
        for record in dataset_records:
            SeqIO.write(record, out_handle, "fasta")
    print(f"Isolated datasets saved to {output_file}")

def main():
    args = parse_args()
    isolate_datasets(args.input_file, args.output_dir, args.datasets)

if __name__ == "__main__":
    main()