import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from misvm import MISVM
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_auc_score, roc_curve,
    make_scorer, balanced_accuracy_score, auc)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from src.pooling_methods import max_pool_embeddings, mean_pool_embeddings
from src.read_cluster import read_cluster
from src.io import load_config
from src.make_mode_embeddings import merge_msa_nmd, make_mode_embeddings

"""
This script performs nested cross-validation with SVC. It trains two baseline models,
one on sequence embeddings and one on structure embeddings, and then two models with
max-pooled and mean-pooled structure embeddings across conformations, a model with mode embeddings,
and a bag-level MI-SVM model.

To use the --cluster option, you need to have a .tsv file with clustered sequences in the specified path,
that was produced using the "mmseqs easy-cluster" command.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Nested CV with SVC on Embeddings")
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--variable', type=str, choices=['class', 'acyclic_status', 'precursor_cation'], default='acyclic_status')
    parser.add_argument('--datasets', type=str, nargs='+', default='sts', help='Dataset to use (default: sts)')
    parser.add_argument('--cluster', action='store_true', help='Use clustered sequences')
    return parser.parse_args()


def main():
    args = parse_args()
    # Load the DataFrame from a pickle file
    df = pd.read_pickle(args.data)
    all_datasets = df['dataset'].unique().tolist()
    for x in all_datasets:
        if type(x) != str:
            all_datasets.remove(x) 
    print(f"Available datasets: {all_datasets}")
    variable = args.variable
    if variable == 'parent_cation':
        df = df.rename(columns={'parent_cation': 'precursor_cation'})
        variable = 'precursor_cation'
    datasets = args.datasets
    if len(datasets) > 1:
        dataset_id = '-'.join(datasets)
    elif len(datasets) == 1:
        dataset_id = datasets[0]
    cluster = args.cluster
    if 'all' in datasets:
        datasets = all_datasets

    # Load the configuration file
    config = load_config()
    duplicate_sequences = config.get('duplicate_sequences', [])
    paths = config.get('paths', {})

    # Check if the output directory exists, if not create it
    output_dir = f"results/machine_learning/{variable}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #df.rename(columns={'embedding': 'sequence_embeddings', 'length': 'class'}, inplace=True)

    # Filter the DataFrame based on the variable, get target
    if variable == 'class':
        df = df[~df['class'].isin(['SESTER-TS','SESQUAR-TS', 'MTS/STS'])].reset_index(drop=True)
    elif variable == 'precursor_cation':
        df = df[df['precursor_cation'].isin(['farnesyl', 'nerolidyl'])].reset_index(drop=True)
    elif variable == 'acyclic_status':
        df = df[df['acyclic_status'].isin(['acyclic', 'cyclic'])].reset_index(drop=True)

    # Filter the DataFrame based on the datasets
    dataset_df = df[df['dataset'].isin(datasets)].reset_index(drop=True)

    # Remove potential duplicate sequences
    dataset_df = dataset_df[~dataset_df['seq_id'].isin(duplicate_sequences)].reset_index(drop=True)

    # Make the mode embeddings
    if 'all' not in datasets:
        if 'mode_embeddings' in dataset_df.columns:
            dataset_df = dataset_df.drop(columns=['mode_embeddings'])
        msa_folder = paths['msa_folder']
        msa_file = f'{msa_folder}/{dataset_id}.aln-clustal_num'
        nmd_folder = paths['nmd_folder']
        mode_df = merge_msa_nmd(msa_file, nmd_folder)
        mode_df = make_mode_embeddings(mode_df)
        # Merge the mode embeddings with the dataset DataFrame
        dataset_df = pd.merge(dataset_df, mode_df, on='seq_id', how='left')

    # If a cluster file is provided, filter the dataset based on the cluster
    if cluster:
        cluster_path = paths['cluster_folder']
        cluster_file = f'{cluster_path}/{dataset_id}/{dataset_id}_cluster.tsv'
        cluster_seq_ids = read_cluster(cluster_file)

        # Create a regex pattern to match any ID in the cluster
        cluster_pattern = '|'.join(map(re.escape, cluster_seq_ids))
        dataset_df = dataset_df[dataset_df['seq_id'].str.contains(cluster_pattern, regex=True)].reset_index(drop=True)

    # Prepare the data for MIL
    seq_ids = dataset_df["seq_id"].unique()
    bags, y_mil = [], []
    for i, seq in enumerate(seq_ids):
        # Get the embeddings for the current sequence
        sub = dataset_df[dataset_df["seq_id"] == seq]
        struct = sub["structure_embeddings"]
        emb = np.vstack(struct.tolist())
        bags.append(emb)

        # Determine the label for the bag based on the variable
        if variable == "acyclic_status":
            lbl_str = sub["acyclic_status"].iloc[0]   # either 'acyclic' or 'cyclic'
            # Choose which you want as positive:
            y_mil.append(+1 if lbl_str == "cyclic" else -1)

        elif variable == "precursor_cation":
            lbl_str = sub[variable].iloc[0]          # 'farnesyl' or 'nerolidyl'
            y_mil.append(+1 if lbl_str == "nerolidyl" else -1)
    y_mil = np.array(y_mil)    

    print("Total bags:", len(y_mil))
    print("Positive bags (+1):", np.sum(y_mil == 1))
    print("Negative bags (-1):", np.sum(y_mil == -1))

    # Set the original dataset DataFrame for the structure baseline
    original_dataset_df = dataset_df[dataset_df["version"] == "v0"].reset_index(drop=True)

    # Prepare the features and labels for the classifiers
    X = original_dataset_df[["sequence_embeddings", "structure_embeddings", "mode_embeddings"]]
    y = original_dataset_df[variable].values

    # Create filtered DataFrame
    filtered_dataframe = original_dataset_df.drop(columns=['structure_embeddings'])

    # Prepare the max and mean pooled datasets
    max_dataset_df = dataset_df.groupby('seq_id')['structure_embeddings'].apply(max_pool_embeddings).reset_index()
    max_dataset_df = pd.merge(max_dataset_df, filtered_dataframe, on='seq_id', how='right', suffixes=('', '_y'))
    X_max = max_dataset_df
    y_max = max_dataset_df[variable].values

    mean_dataset_df = dataset_df.groupby('seq_id')['structure_embeddings'].apply(mean_pool_embeddings).reset_index()
    mean_dataset_df = pd.merge(mean_dataset_df, filtered_dataframe, on='seq_id', how='right', suffixes=('', '_y'))
    X_mean = mean_dataset_df
    y_mean = mean_dataset_df[variable].values

    # Encode the labels
    le = LabelEncoder()
    le_max = LabelEncoder()
    le_mean = LabelEncoder()
    y = le.fit_transform(original_dataset_df[variable].values)
    y_max = le_max.fit_transform(y_max)
    y_mean = le_mean.fit_transform(y_mean)
    classes = le.classes_
    print(f"Classes: {classes}")
    print(f'Positive class: {classes[1]}')

    # Print the ratio of positive to negative samples
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    print(f"Positive samples: {pos_count}, Negative samples: {neg_count}")

    # Create a custom scorer for AUC using predict_proba
    roc_auc_proba_scorer = make_scorer(
        balanced_accuracy_score
        #needs_proba=True
        )

    # Create a custom SVC model class for each type of embedding
    class NormalSVCModel(BaseEstimator, ClassifierMixin):
        def __init__(self, col, est):
            self.col = col
            self.est = est

        def fit(self, X, y):
            X_emb = np.vstack(X[self.col].values)
            self.est.fit(X_emb, y)
            return self

        def predict(self, X):
            X_emb = np.vstack(X[self.col].values)
            return self.est.predict(X_emb)

        def predict_proba(self, X):
            X_emb = np.vstack(X[self.col].values)
            return self.est.predict_proba(X_emb)
        
        def decision_function(self, X):
            X_emb = np.vstack(X[self.col].values)
            return self.est.decision_function(X_emb)
    
    # Create custom SVC models for max and mean pooled embeddings
    class MaxPooledSVCModel(BaseEstimator, ClassifierMixin):
        def __init__(self, col, est):
            self.col = col
            self.est = est

        def fit(self, X_max, y):
            X_emb = np.vstack(X_max[self.col].values)
            self.est.fit(X_emb, y)
            return self

        def predict(self, X_max):
            X_emb = np.vstack(X_max[self.col].values)
            return self.est.predict(X_emb)

        def predict_proba(self, X_max):
            X_emb = np.vstack(X_max[self.col].values)
            return self.est.predict_proba(X_emb)
        
        def decision_function(self, X_max):
            X_emb = np.vstack(X_max[self.col].values)
            return self.est.decision_function(X_emb)
        
    class MeanPooledSVCModel(BaseEstimator, ClassifierMixin):
        def __init__(self, col, est):
            self.col = col
            self.est = est

        def fit(self, X_mean, y):
            X_emb = np.vstack(X_mean[self.col].values)
            self.est.fit(X_emb, y)
            return self

        def predict(self, X_mean):
            X_emb = np.vstack(X_mean[self.col].values)
            return self.est.predict(X_emb)

        def predict_proba(self, X_mean):
            X_emb = np.vstack(X_mean[self.col].values)
            return self.est.predict_proba(X_emb)
        
        def decision_function(self, X_mean):
            X_emb = np.vstack(X_mean[self.col].values)
            return self.est.decision_function(X_emb)

        
    # Set the models for each type of embedding
    models = {
        "sequence_classifier": NormalSVCModel(
            "sequence_embeddings",
            SVC(probability=True, class_weight="balanced")),
        "structure_classifier": NormalSVCModel(
            "structure_embeddings",
            SVC(probability=True, class_weight="balanced")),
        "mode_classifier": NormalSVCModel(
            "mode_embeddings",
            SVC(probability=True, class_weight="balanced")),
        "bag_conformation_max_classifier": MaxPooledSVCModel(
            "structure_embeddings",
            SVC(probability=True, class_weight="balanced")),
        "bag_conformation_mean_classifier": MeanPooledSVCModel(
            "structure_embeddings",
            SVC(probability=True, class_weight="balanced"))
        }


    # --- 4. Hyperparameter grids (you can adjust these)
    param_grids = {
        "sequence_classifier": {
            "est__C": [0.1, 1, 10],
            "est__gamma": ["scale", "auto"]
        },
        "structure_classifier": {
            "est__C": [0.1, 1, 10],
            "est__gamma": ["scale", "auto"]
        },
        "mode_classifier": {
            "est__C": [0.1, 1, 10],
            "est__gamma": ["scale", "auto"]
        },
        "bag_conformation_max_classifier": {
            "est__C": [0.1, 1, 10],
            "est__gamma": ["scale", "auto"]
        },
        "bag_conformation_mean_classifier": {
            "est__C": [0.1, 1, 10],
            "est__gamma": ["scale", "auto"]
        }
    }

    # Set up outer and inner CV (stratified for imbalanced labels)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


    # Storage for per‐fold ROC data
    results = {
        name: {
            "fprs": [],
            "tprs": [],
            "aucs": [], "y_true": [], "y_pred": []
        }
        for name in models }
    results["bag_mi_svm"] = { "fprs": [], "tprs": [], "aucs": [], "y_true": [], "y_pred": []}
    

    # The nested‐CV loop
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_max_train, X_max_test = X_max.iloc[train_idx], X_max.iloc[test_idx]
        y_max_train, y_max_test = y_max[train_idx], y_max[test_idx]
        X_mean_train, X_mean_test = X_mean.iloc[train_idx], X_mean.iloc[test_idx]
        y_mean_train, y_mean_test = y_mean[train_idx], y_mean[test_idx]


        for name, model in models.items():
            gs = GridSearchCV(
                    estimator = model,
                    param_grid = param_grids[name],
                    cv = inner_cv,
                    scoring = roc_auc_proba_scorer, 
                    n_jobs = -1,
                    refit = True
                )
            
            # Select the appropriate training set based on the model
            if name == "sequence_classifier" or name == "structure_classifier" or name == "mode_classifier":
            # Tune on the outer‐training set
                gs.fit(X_train, y_train)

                # Evaluate on the outer‐test set
                y_pred  = gs.predict(X_test)
                y_proba = gs.predict_proba(X_test)[:, 1]

                # Compute ROC for this fold
                fpr_fold, tpr_fold, _ = roc_curve(y_test, y_proba)
                auc_fold = roc_auc_score(y_test, y_proba)
                results[name]["fprs"].append(fpr_fold)
                results[name]["tprs"].append(tpr_fold)
                results[name]["aucs"].append(auc_fold)
                yt = np.where(y_test == 1,  1, -1)
                yp = np.where(y_pred   == 1,  1, -1)
                results[name]["y_true"].extend(yt.tolist())
                results[name]["y_pred"].extend(yp.tolist())


            
            elif name == "bag_conformation_max_classifier":
                # Tune on the outer‐training set
                gs.fit(X_max_train, y_max_train)

                # Evaluate on the outer‐test set
                y_pred  = gs.predict(X_max_test)
                y_proba = gs.predict_proba(X_max_test)[:, 1]

                # Compute ROC for this fold
                fpr_fold, tpr_fold, _ = roc_curve(y_max_test, y_proba)
                auc_fold = roc_auc_score(y_max_test, y_proba)
                results[name]["fprs"].append(fpr_fold)
                results[name]["tprs"].append(tpr_fold)
                results[name]["aucs"].append(auc_fold)
                yt = np.where(y_max_test == 1,  1, -1)
                yp = np.where(y_pred   == 1,  1, -1)
                results[name]["y_true"].extend(yt.tolist())
                results[name]["y_pred"].extend(yp.tolist())

            elif name == "bag_conformation_mean_classifier":
                # Tune on the outer‐training set
                gs.fit(X_mean_train, y_mean_train)

                # Evaluate on the outer‐test set
                y_pred  = gs.predict(X_mean_test)
                y_proba = gs.predict_proba(X_mean_test)[:, 1]


                # Compute ROC for this fold
                fpr_fold, tpr_fold, _ = roc_curve(y_mean_test, y_proba)
                auc_fold = roc_auc_score(y_mean_test, y_proba)
                results[name]["fprs"].append(fpr_fold)
                results[name]["tprs"].append(tpr_fold)
                results[name]["aucs"].append(auc_fold)
                yt = np.where(y_mean_test == 1,  1, -1)
                yp = np.where(y_pred   == 1,  1, -1)
                results[name]["y_true"].extend(yt.tolist())
                results[name]["y_pred"].extend(yp.tolist())
    
        #print(results)
        Cs = [0.1, 1, 10]  # hyperparameter grid for MI-SVM

        mean_fpr   = np.linspace(0, 1, 100)
        all_interp = []
        all_aucs   = []

        for outer_train, outer_test in outer_cv.split(bags, y):
            # split bags
            bags_tr = [bags[i] for i in outer_train]
            y_tr    = y[outer_train]
            bags_te = [bags[i] for i in outer_test]
            y_te    = y[outer_test]

        bags_tr = [bags[i] for i in train_idx]
        y_tr    = y_mil[train_idx]
        bags_te = [bags[i] for i in test_idx]
        y_te    = y_mil[test_idx]


        # 4) manual inner-loop grid search
        best_C, best_score = None, -np.inf
        for C in Cs:
            inner_aucs = []
            for inner_train, inner_val in inner_cv.split(bags_tr):
                # train on inner_train
                m = MISVM(kernel="linear", C=C, max_iters=50)
                m.fit([bags_tr[i] for i in inner_train],
                    y_tr[inner_train])

                # score on inner_val
                scores_val = m.predict([bags_tr[i] for i in inner_val])
                yv_bin     = (y_tr[inner_val] == 1).astype(int)
                fpr, tpr, _ = roc_curve(yv_bin, scores_val)
                inner_aucs.append(auc(fpr, tpr))

            mean_inner_auc = np.mean(inner_aucs)
            if mean_inner_auc > best_score:
                best_score, best_C = mean_inner_auc, C

        # Retrain on full outer-train with best_C
        final_m = MISVM(kernel="linear", C=best_C, max_iters=20)
        final_m.fit(bags_tr, y_tr)
        scores_te = final_m.predict(bags_te)
        print(f"Fold bag‐scores:  min={scores_te.min():.3f},  max={scores_te.max():.3f}")


        # Threshold at 0 to get discrete ±1 labels
        y_pred_mi = np.where(scores_te > 0, 1, -1)


        # Compute ROC using the raw scores
        fpr, tpr, _ = roc_curve((y_te == 1).astype(int), scores_te)
        fold_auc    = auc(fpr, tpr)
        # Interpolate & store
        print(f"Outer fold AUC={fold_auc:.3f} (best C={0.1})")


        results["bag_mi_svm"]["fprs"].append(fpr)
        results["bag_mi_svm"]["tprs"].append(tpr)
        results["bag_mi_svm"]["aucs"].append(fold_auc)
        results["bag_mi_svm"]["y_true"].extend(y_te.tolist())
        results["bag_mi_svm"]["y_pred"].extend(y_pred_mi.tolist())

    plt.figure(figsize=(8,6))

    # Common grid for interpolation:
    mean_fpr = np.linspace(0, 1, 100)

    # Plot mean ROC curves for each model
    for name, res in results.items():
        # Interpolate each fold's TPR onto mean_fpr
        interp_tprs = []
        for fpr_fold, tpr_fold in zip(res["fprs"], res["tprs"]):
            interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(interp_tprs, axis=0)
        mean_auc = np.mean(res["aucs"])
        std_auc = np.std(res["aucs"])

        plt.plot(
            mean_fpr, mean_tpr,
            label=f"{name} (AUC={mean_auc:.2f}±{std_auc:.2f})"
        )
        plt.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            alpha=0.2
        )

    plt.plot([0,1], [0,1], linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC Curves for {variable} {'(Clustered)' if cluster else ''}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if cluster:
        plt.savefig(f"results/machine_learning/{variable}/mean_roc_curves_{variable}_clustered.png", dpi=150)
    else:
        plt.savefig(f"results/machine_learning/{variable}/mean_roc_curves_{variable}.png", dpi=150)
    plt.show()

    # Print classification reports and confusion matrices
    for name, res in results.items():
        y_true = np.array(res["y_true"])
        y_pred = np.array(res["y_pred"])

        unique = np.unique(np.concatenate([y_true, y_pred]))
        print(f"{name}: found labels {unique}")

        cm = confusion_matrix(y_true, y_pred, labels=[1, -1])
        print(f"\n=== {name} ===")
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

        if variable == 'precursor_cation':
            classes = ["nerolidyl", "farnesyl"]
        elif variable == 'acyclic_status':
            classes = ["acyclic", "cyclic"]

        # Plot classification report
        report_dict = classification_report(
            y_true, y_pred,
            labels=[1, -1],
            target_names=classes,
            zero_division=0            
        )

        df_report = pd.DataFrame(report_dict).T

        df_report = df_report.round(2)

        fig, ax = plt.subplots(figsize=(6, 2 + 0.5 * len(df_report)))
        ax.axis('off')

        tbl = ax.table(
            cellText=df_report.values,
            colLabels=df_report.columns,
            rowLabels=df_report.index,
            loc='center',
            cellLoc='center'
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.5)

        fig.tight_layout()
        if cluster:
            fig.savefig(f"results/machine_learning/{variable}/classification_report_{name}_{variable}_clustered.png", dpi=150)
        else:
            fig.savefig(f"results/machine_learning/{variable}/classification_report_{name}_{variable}.png", dpi=150)
        plt.close(fig)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix"
        )

        # Annotate each cell
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        if cluster:
            plt.savefig(f"results/machine_learning/{variable}/new_confusion_matrix_{name}_{variable}_clustered.png", dpi=150)
        else:
            plt.savefig(f"results/machine_learning/{variable}/new_confusion_matrix_{name}_{variable}.png", dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    main()