import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.pooling_methods import max_pool_embeddings, mean_pool_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Nested CV with SVC on Embeddings")
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--variable', type=str, choices=['class', 'acyclic_status', 'precursor_cation'], default='acyclic_status')
    parser.add_argument('--feature', type=str , default='structure_embeddings',)
    parser.add_argument('--datasets', type=str, nargs='+', default='sts', help='Dataset to use (default: sts)')
    parser.add_argument('--pool', type=str, choices=['max', 'mean', None], default=None, help='Pooling method for embeddings (default: None)')
    return parser.parse_args()

def plot_pca(data, variable, datasets, feature, pool_method=None):
    # Filter data for the specified datasets
    data = data[data['dataset'].isin(datasets)]
    if not pool_method:
        data = data[data['version'] == 'v0']
    elif pool_method == 'max':
        if pool_method == 'max':
            pooled_data = data.groupby('seq_id')['structure_embeddings'].apply(max_pool_embeddings).reset_index()
        elif pool_method == 'mean':
            pooled_data = data.groupby('seq_id')['structure_embeddings'].apply(mean_pool_embeddings).reset_index()
        data = data.drop(columns=['structure_embeddings'])
        data = data[data['version'] == 'v0']
        data = data.merge(pooled_data, on='seq_id', how='right')
        print(data.shape)
    if variable == 'acyclic_status':
        data = data[data['acyclic_status'].isin(['acyclic', 'cyclic'])]
    elif variable == 'parent_cation' or variable == 'precursor_cation':
        variable = 'precursor_cation'
        data.rename(columns={'parent_cation': 'precursor_cation'}, inplace=True)
        data = data[data['precursor_cation'].isin(['nerolidyl', 'farnesyl'])]

    X = np.vstack(data[feature].values)
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(X)
    data = data.copy()
    data['PCA1'], data['PCA2'] = emb[:,0], emb[:,1]

    # Calculate the explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PCA components: {explained_variance}")

    # get unique datasets and variable labels
    ds_list = list(data['dataset'].unique())
    var_list = list(data[variable].unique())

    # pick some markers (extend this list if you have >10 datasets)
    markers = ['o','s','^','D','v','P','X','<','>','*']
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(ds_list)}

    # pick colors for each variable label
    cmap = plt.get_cmap('Dark2', len(var_list))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(var_list)}

    # plot
    fig, ax = plt.subplots(figsize=(8,6))
    for ds in ds_list:
        for lbl in var_list:
            sub = data[(data['dataset']==ds) & (data[variable]==lbl)]
            if sub.empty:
                continue
            ax.scatter(
                sub['PCA1'], sub['PCA2'],
                marker=marker_map[ds],
                color=color_map[lbl],
                s=30, alpha=0.7,
                label=f"{ds} / {lbl}"
            )

    ax.set_xlabel(f"PCA1 \n (Explained Variance: {explained_variance[0]:.2f})")
    ax.set_ylabel(f"PCA2 \n (Explained Variance: {explained_variance[1]:.2f})")
    title_pool = f", Pooling={pool_method}" if pool_method else ""
    ax.set_title(f"{feature} ▶︎ PCA by Dataset{title_pool}")
    # one legend for all combinations
    ax.legend(
        title=f"{variable} (color) \n  dataset (shape)",
        bbox_to_anchor=(1.05,1), loc="upper left"
    )
    plt.tight_layout()

    # save & show
    out_dir = f"results/{variable}"
    os.makedirs(out_dir, exist_ok=True)
    pool_ext = f"_{pool_method}" if pool_method else ""
    plt.savefig(f"{out_dir}/pca_{feature}{pool_ext}.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    args = parse_args()
    
    # Load the data
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} does not exist.")
    
    data = pd.read_pickle(args.data)
    
    # Rename the 'embedding' column to 'sequence_embeddings' if it exists
    if 'embedding' in data.columns:
        data.rename(columns={'embedding': 'sequence_embeddings'}, inplace=True)

    
    # Check if the specified feature exists in the data
    if args.feature not in data.columns:
        raise ValueError(f"Feature '{args.feature}' not found in the data.")
    
    # Plot UMAP
    plot_pca(data, args.variable, args.datasets, args.feature, pool_method=args.pool)

if __name__ == "__main__":
    main()