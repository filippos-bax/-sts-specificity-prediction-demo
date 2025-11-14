import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.manifold import MDS

def parse_args():
    parser = argparse.ArgumentParser(description='Plot MDS from a pickle file.')
    parser.add_argument('--data', type=str, help='Path to the pickle file containing features')
    parser.add_argument('--matrix', type=str, help='Matrix to plot MDS for')
    parser.add_argument('--variable', type=str, help='Variable to plot MDS for')
    parser.add_argument('--datasets', type=str, nargs='+', help='List of datasets to include in the plot')
    return parser.parse_args()

def plot_mds(data, matrix, variable, datasets):
    # Extract the specified matrix and feature
    data = data[data['version'] == 'v0']
    data = data[data['dataset'].isin(datasets)]
    if variable == 'precursor_cation':
        data = data[data[variable].isin(['nerolidyl', 'farnesyl'])]
    elif variable == 'acyclic_status':
        data = data[data[variable].isin(['acyclic', 'cyclic'])]
    data = data.set_index('seq_id', drop=True)

    # Perform MDS
    mds = MDS(n_components=2, random_state=42, dissimilarity='precomputed')
    mds_result = mds.fit_transform(matrix)

    # Create a DataFrame for the MDS results
    mds_df = pd.DataFrame(mds_result, columns=['MDS1', 'MDS2'], index=matrix.index)

    plot_df = pd.merge(data, mds_df, left_index=True, right_index=True)

    # Plot the MDS results
    # get unique datasets and variable labels
    ds_list = list(plot_df['dataset'].unique())
    var_list = list(plot_df[variable].unique())

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
            sub = plot_df[(plot_df['dataset']==ds) & (plot_df[variable]==lbl)]
            if sub.empty:
                continue
            ax.scatter(
                sub['MDS1'], sub['MDS2'],
                marker=marker_map[ds],
                color=color_map[lbl],
                s=30, alpha=0.7,
                label=f"{ds} / {lbl}"
            )

    ax.set_xlabel("MDS1")
    ax.set_ylabel(f"MDS2")
    ax.set_title(f"{variable} ▶︎ MDS")
    # one legend for all combinations
    ax.legend(
        title=f"{variable} (color) \n  dataset (shape)",
        bbox_to_anchor=(1.05,1), loc="upper left"
    )
    plt.tight_layout()

    # save & show
    out_dir = f"results/{variable}"
    plt.savefig(f"{out_dir}/{variable}_mds.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def main():
    args = parse_args()
    data = args.data
    matrix = args.matrix
    variable = args.variable
    datasets = args.datasets
    # Load the data
    data = pd.read_pickle(data)
    matrix = pd.read_table(matrix, skiprows=1, header=None, sep='\s+', index_col=0)
    print(matrix.index)

    if variable == 'parent_cation':
        variable = 'precursor_cation'
        data.rename(columns={'parent_cation': 'precursor_cation'}, inplace=True)
    plot_mds(data, matrix, variable, datasets)

if __name__ == "__main__":
    main()