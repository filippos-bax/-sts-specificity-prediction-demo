# sts-specificity-prediction
The scripts that use the prody and geometricus packages were run with Python 3.10. Everything else was run with Python 3.12.

# Make a combined metadata DataFrame from all of the individual datasets
python -m scripts.make_metadata

# Isolate datasets from the combined fasta file and make new fasta
python -m scripts/isolate_datasets --input_file data/processed/03-combined-with-identifiers/combined.fasta \
	--output_dir data/processed/03-combined-with-identifiers/ \
	--datasets sts mts

# Generate sequence embeddings
python scripts/create_sequence_embeddings.py

# Generate structures
sbatch bash_scripts/run_alphafold.sh

# Create structure embeddings and merge to metadata
python scripts/create_structure_embeddings.py --i data/processed/05-structures --o data/processed/06-structure-embeddings
python scripts/join_structure_embeddings.py

# Run NMA and sample conformations
sbatch bash_scripts/nma.sh

# Create structure embeddings for all conformations and merge to metadata
python scripts/create_structure_embeddings.py --i data/processed/08-conformations --o data/processed/09-conformation-embeddings
python scripts/join_conformation_embeddings.py --input_meta data/processed/06-structure-embeddings/metadata_with_structure_embeddings.pkl \
	--input_embeddings data/processed/09-conformation-embeddings/structure_embeddings.csv

# Create mode embeddings for the entire dataset
python -m scripts.create_mode_embeddings --meta data/processed/09-conformation-embeddings/metadata_with_conformation_embeddings.pkl

# Create clusters
mmseq easy-cluster \
  data/processed/03-combined-with-identifiers/combined.fasta \
  data/processed/10-mmseqs/all \
  data/processed/xx-temp/ 


# Run the classifier 
python -m scripts.machine_learning_classifier --data data/processed/12-mode-embeddings/metadata_with_mode_embeddings.pkl \
	--variable precursor_cation \
	--datasets sts \
	--clustered
	
# Make mds plot
python scripts/plot_mds.py --data --data data/processed/12-mode-embeddings/metadata_with_mode_embeddings.pkl \
	--matrix data/processed/11-msas/sts-mts.matrix
	--variable precursor_cation
	--datasets sts mts
	
# Make UMAP plot
python -m scripts.plot_umap --data --data data/processed/12-mode-embeddings/metadata_with_mode_embeddings.pkl \
	--variable precursor_cation
	--feature sequence_embeddings
	--datasets sts mts
	
# Make PCA plot
python -m scripts.plot_pca --data --data data/processed/12-mode-embeddings/metadata_with_mode_embeddings.pkl \
	--variable precursor_cation
	--feature sequence_embeddings
	--datasets sts mts
