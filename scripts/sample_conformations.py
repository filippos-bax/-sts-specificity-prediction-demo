from prody import *
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

"""
#########################################
RUN THIS SCRIPT WITH PYTHON 3.10 OR LOWER
#########################################

This script performs Normal Mode Analysis (NMA) on a SINGLE protein structure, 
using the ANM (Anisotropic Network Model) method. 
It samples a set of conformations based on the normal modes and saves them to PDB files.

THE SCRIPT REQUIRES A DSSP FILE IN ORDER TO TRIM THE TERMINAL LOOP REGIONS OF THE PROTEIN.
USE mkdssp TO GENERATE A DSSP FILE FROM THE PDB FILE BEFORE RUNNING THIS SCRIPT.
"""


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process PDB and DSSP files for ANM analysis.')

    # Required arguments 
    parser.add_argument('--pdb', type=str, required=True, help='Path to the PDB file')
    parser.add_argument('--dssp', type=str, required=True, help='Path to the DSSP file')

    # Required output directory
    parser.add_argument('--pdb_outdir', type=str, default='.', required=True, help='Directory to save pdb file')

    # Optional arguments
    parser.add_argument('--nmd_outdir', type=str, help='Directory to save NMD file. If not specified, will save file to PDB outdir.')
    parser.add_argument('--png_outdir', type=str, help='Directory to save PNG files. If not specified, will not save PNG files.')
    parser.add_argument('--modes', type=int, default=5, help='Number of modes to calculate (default: 5)')
    parser.add_argument('--n_confs', type=int, default=40, help='Number of conformations to be generated (default: 40). \\n' \
    'The reference protein is also generated, after being trimmed.')
    parser.add_argument('--rmsd', type=int, default=1, help='Step size for generating deformed structures (default: 1)')

    return parser.parse_args()

# Resolve command line arguments
def resolve_output_paths(args):
    
    pdb_outdir = args.pdb_outdir

    # Ensure the nmd outdir is set, if not use the pdb outdir
    if not args.nmd_outdir:
        nmd_outdir = pdb_outdir
    else:
        nmd_outdir = args.nmd_outdir

    
    return pdb_outdir, nmd_outdir


def main():

    # Parse and resolve command line arguments
    args = parse_args()
    pdb_outdir, nmd_outdir = resolve_output_paths(args)
    n_confs = args.n_confs
    if n_confs < 0:
        raise ValueError("Number of conformations must be a positive number.")
    rmsd = args.rmsd


    # Define the pdb file path
    pdb_file = args.pdb

    # Define the DSSP file path
    dssp_file = args.dssp

    # Get the protein name from the file path directory
    protein_name = os.path.splitext(os.path.basename(pdb_file))[0]

    # Ensure the output directories exist
    if not os.path.exists(pdb_outdir):
        raise FileNotFoundError(f"The specified PDB output directory '{pdb_outdir}' does not exist.")
    if nmd_outdir and not os.path.exists(nmd_outdir):
        raise FileNotFoundError(f"The specified NMD output directory '{nmd_outdir}' does not exist.")
    if args.png_outdir and not os.path.exists(args.png_outdir):
        raise FileNotFoundError(f"The specified PNG output directory '{args.png_outdir}' does not exist.")

    # Load the pdb file
    protein = parsePDB(pdb_file)

    # Check if protein was loaded correctly
    if protein is None:
        raise ValueError(f"Failed to load the protein from {pdb_file}")

    # Load the DSSP file
    protein = parseDSSP(dssp_file, protein)

    # Set protein length
    protein_length = protein.numAtoms('calpha')

    # Remove non-helix regions from the start of the protein
    for i in range (1, protein_length+1):
        sec_str = protein.getSecstrs()
        if str(sec_str[0]) != 'H':
            protein = protein.select(f'resid {i+1} to {protein_length}')
        else:
            break

    # Set new starting residue
    start_res = protein.getResnums()[0]

    # Remove non-helix regions from the end of the protein
    for i in range (1, protein_length + 1):
        sec_str = protein.getSecstrs()
        if str(sec_str[-1]) != 'H':
            protein = protein.select(f'resid {start_res} to {protein_length - i}')
        else:
            break

    protein = protein.copy()

    # Select the C-alpha atoms
    protein_ca = protein.ca

    # Perform Anisotropic Network Model (ANM) normal mode analysis
    modes=args.modes
    anm = ANM('Analysis')
    anm.buildHessian(protein_ca)
    anm.calcModes(n_modes=modes)    

    if args.png_outdir:
        for i in range (0, modes):
            # Show the first mode
            showMode(anm[i])

            # Change the title of the plot
            plt.title(f'Mode {i+1} from ANM analysis of {protein_name}')

            # Save the plot instead of showing it
            plt.savefig(f'{args.png_outdir}/{protein_name}_normal_mode_{i+1}.png')
            print(f"Plot saved as '{protein_name}_normal_mode_{i+1}.png'. You can download it to view.")

            # Clear the current plot to avoid overlap
            plt.clf

        # Plot mobility of residues
        showSqFlucts(anm)

        # Change the title of the plot
        plt.title(f'Square Fluctuations of {protein_name}')

        # Save the plot instead of showing it
        plt.savefig(f'{args.png_outdir}/{protein_name}_square_fluctuations_helix.png')

        # Clear the current plot to avoid overlap
        plt.clf()


    # Save the protein structure to a PDB file
    writePDB(f'{pdb_outdir}/{protein_name}*v0.pdb', protein, secondary=False)

    # Save to NMD file
    writeNMD(f'{nmd_outdir}/{protein_name}.nmd', anm[:modes], atoms=protein_ca)

    # Extend the anm
    anm_ext, protein_all = extendModel(anm, protein_ca, protein, norm=True)

    if args.png_outdir:
        # Show the extended model
        showSqFlucts(anm_ext)

        # Change the title of the plot
        plt.title(f'Extended Model of {protein_name}')

        # Save the plot instead of showing it
        plt.savefig(f'{args.png_outdir}/{protein_name}_extended_model_fluctuations.png')

        # Clear the current plot to avoid overlap
        plt.clf()

    # Sample the conformations to make the ensemble
    ens = sampleModes(anm_ext, atoms=protein.protein, n_confs=n_confs, rmsd=rmsd)

    if args.png_outdir:
        # Write the ensemble to DCD file
        writeDCD(f'{nmd_outdir}/{protein_name}.dcd' , ens)

        # Show projection of the ensemble in the mode space
        showProjection(ens, anm_ext[:3], rmsd=True)

        # Change the title of the plot
        plt.title(f'Projection of Ensemble in Mode Space for {protein_name}')

        # Save the plot instead of showing it
        plt.savefig(f'{args.png_outdir}/{protein_name}_ensemble_projection.png')

        # Clear the current plot to avoid overlap
        plt.clf()

        # Get the RMSD values of the ensemble
        rmsd_values = ens.getRMSDs()

        # Plot the RMSD values in a histogram
        plt.hist(rmsd_values, bins=30, color='blue', alpha=0.7)
        plt.xlabel('RMSD (Å)')
        plt.ylabel('Frequency')
        plt.title(f'RMSD Distribution of Ensemble for {protein_name}')
        plt.grid(True)
        # Save the plot instead of showing it
        plt.savefig(f'{args.png_outdir}/{protein_name}_ensemble_rmsd_distribution.png')
        # Clear the current plot to avoid overlap
        plt.clf()

    # Add conformations to AtomGroup object
    protein.addCoordset(ens.getCoordsets())

    # Change the beta values
    protein.all.setBetas(0)
    protein.ca.setBetas(1)

    # Write out the conformations to PDB files
    for i in range(1, protein.numCoordsets()):
        writePDB(f'{pdb_outdir}/{protein_name}*v{i}.pdb', protein, csets=i, secondary=False)
        print(f"Conformation {i} saved as '{protein_name}*v{i}.pdb'.")

if __name__ == "__main__":
    main()

