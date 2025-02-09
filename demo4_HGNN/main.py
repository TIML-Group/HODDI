# Set the environment variable
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import libraries
import ast
import copy
import torch
import config
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CHGNN import CHGNN
from CLOSEgaps import CLOSEgaps
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from itertools import chain
import torch.nn.functional as F
from utils import set_random_seed, create_neg_rxns, getGipKernel, monitor_gradients, extract_chemicals
from algorithms.smiles2vec import smiles_to_chemberta_vector, smiles_to_chemberta_vector_gpu
import GEM_process as GEM

# Clean GPU cache
torch.cuda.empty_cache()  
torch.cuda.ipc_collect()  


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_pre(feature, incidence_matrix, model):
    """
    Perform prediction using the given model and input features.

    Parameters
    ----------
    feature : torch.Tensor
        The input feature tensor for the model.
    incidence_matrix : torch.Tensor
        The incidence matrix representing the connections in the data.
    model : torch.nn.Module
        The model to use for prediction.

    Returns
    -------
    torch.Tensor
        The predicted output tensor, with squeezed dimensions.
    """

    model.eval()
    with torch.no_grad():
        y_pred = model.predict(feature, incidence_matrix)
    return torch.squeeze(y_pred)

def init_weights(m):
    """
    Initialize the weights of a linear layer using Xavier uniform initialization.

    Parameters
    ----------
    m : nn.Module
        The module to initialize.

    Returns
    -------
    None

    Notes
    -----
    This function is used as an initialization function for PyTorch nn.Module objects.
    It initializes the weights using Xavier uniform initialization and the bias using zeros.
    """

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def compute_hyperedge_features(train_incidence_pos, extra_feature_t):

    """
    Compute hyperedge features using XOR operation.

    Parameters
    ----------
    train_incidence_pos : torch.Tensor
        The incidence matrix of positive samples, with shape (num_drugs, num_hyperedges).
    extra_feature_t : torch.Tensor
        The extra feature tensor, with shape (num_drugs, feature_dim).

    Returns
    -------
    hyperedge_features : torch.Tensor
        The computed hyperedge feature tensor, with shape (num_hyperedges, feature_dim).

    Notes
    -----
    This function computes the hyperedge features by performing an XOR operation on the extra features of all drugs in the same hyperedge.
    The result is a new feature tensor with the same shape as the input.
    """
    num_hyperedges = train_incidence_pos.shape[1]  
    feature_dim = extra_feature_t.shape[1]  
    
    hyperedge_features = torch.zeros((num_hyperedges, feature_dim), dtype=extra_feature_t.dtype, device=extra_feature_t.device)

    for e in range(num_hyperedges):
        drug_indices = torch.where(train_incidence_pos[:, e] > 0)[0]
        
        if len(drug_indices) > 0:
            drug_features = extra_feature_t[drug_indices]
            drug_features_int = (drug_features * 1000).to(torch.int32)

            hyperedge_feature_int = drug_features_int[0]
            for i in range(1, len(drug_features_int)):
                hyperedge_feature_int ^= drug_features_int[i]

            hyperedge_feature = hyperedge_feature_int.float() / 1000

            hyperedge_features[e] = hyperedge_feature

    return hyperedge_features

def train(args, X_smiles, incidence_matrix_pos, train_incidence_pos, incidence_train, incidence_valid, y_train, y_valid, iteration):
    """
    Train the CLOSEgaps model using the given dataset and parameters.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing hyperparameters and settings for training.
    X_smiles : list of str
        A list of SMILES strings representing chemical compounds.
    incidence_matrix_pos : torch.Tensor
        The positive incidence matrix, with shape (num_drugs, num_hyperedges).
    train_incidence_pos : torch.Tensor
        The incidence matrix of positive samples for training.
    incidence_train : torch.Tensor
        The incidence matrix for training, with shape (num_drugs, num_samples).
    incidence_valid : torch.Tensor
        The incidence matrix for validation, with shape (num_drugs, num_samples).
    y_train : torch.Tensor
        The labels for the training set, with shape (num_samples,).
    y_valid : torch.Tensor
        The labels for the validation set, with shape (num_samples,).
    iteration : int
        The iteration number for training.

    Returns
    -------
    torch.nn.Module
        The trained model with the best weights based on validation F1 score.

    Notes
    -----
    This function performs training of the CLOSEgaps model over a specified number
    of epochs, utilizing the AdamW optimizer and cross-entropy loss. It includes
    functionality for early stopping based on the validation F1 score and saves
    the best model weights. The training and validation losses are plotted and saved
    as a PNG file.
    """

    node_num, _ = incidence_train.shape
    extra_feature_t = None
    edges_feature_t = None

    if X_smiles:
        print('---------------- calculating smiles2vec feature ------------------------------')
        extra_feature_t = smiles_to_chemberta_vector_gpu(smiles=X_smiles, batch_size=args.s2m_batch_size).to(device)
        edges_feature_t = compute_hyperedge_features(incidence_matrix_pos, extra_feature_t)

    model = CLOSEgaps(algorithm=args.algorithm,
                      input_num=node_num,
                      input_feature_num=train_incidence_pos.shape[1],
                      incidence_matrix_pos=incidence_matrix_pos,
                      extra_feature=extra_feature_t,
                      reaction_feature=edges_feature_t,
                      emb_dim=args.emb_dim, conv_dim=args.conv_dim,
                      head=args.head, p=args.p, L=args.L,
                      use_attention=True, enable_hygnn=True).to(device)

    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crossentropyloss = nn.CrossEntropyLoss()

    max_valid_f1 = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    torch.autograd.set_detect_anomaly(True)
    print('---------------- start training ------------------------------')
    losses = []
    for _ in tqdm(range(args.epoch)):
        model.train()
        epoch_loss = 0
        epoch_size = incidence_train.shape[1] // args.batch_size
        for e in range(epoch_size):
            optimizer.zero_grad()
            y_pred = model(train_incidence_pos, incidence_train[:, e * args.batch_size:(e + 1) * args.batch_size])
            # Check if y_pred contains NaN
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print("Warning: y_pred contains NaN or Inf before loss calculation!")
                continue
            if torch.isnan(y_train).any() or torch.isinf(y_train).any():
                print("Warning: y_train contains NaN or Inf!")
                print(y_train)
            loss = crossentropyloss(y_pred, y_train[e * args.batch_size:(e + 1) * args.batch_size])
            loss.backward()
            
            optimizer.step()
            # Accumulate loss
            epoch_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = epoch_loss / epoch_size
        losses.append(avg_loss)

        valid_score = test_pre(train_incidence_pos, incidence_valid, model)

        true_valid_score = valid_score.cpu().numpy()[:, 1]

        y_valid = y_valid.cpu()
        auc_score = roc_auc_score(y_valid, true_valid_score)

        b_score = [int(s >= 0.5) for s in true_valid_score]

        pr = precision_score(y_valid, b_score, zero_division=0)
        re = recall_score(y_valid, b_score)
        f1 = f1_score(y_valid, b_score)
        aupr = average_precision_score(y_valid, true_valid_score)
        if max_valid_f1 < f1:
            max_valid_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'\nvalid, epoch:{_}, f1:{f1},pr:{pr},recall:{re},auc:{auc_score},aupr:{aupr}')
    
    # Plot the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, args.epoch + 1), losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'./output/loss_curve_{iteration}.png', format='png', dpi=300)  
    plt.close()  
    model.load_state_dict(best_model_wts)
    return model

def predict(model, train_incidence_pos, incidence_test, y_test):
    """
    Make predictions on the test set using the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The CLOSEgaps model to use for prediction.
    train_incidence_pos : torch.Tensor
        The incidence matrix of positive samples for training.
    incidence_test : torch.Tensor
        The incidence matrix for testing.
    y_test : torch.Tensor
        The labels for the test set.

    Returns
    -------
    f1 : float
        The F1 score of the model on the test set.
    pr : float
        The precision of the model on the test set.
    re : float
        The recall of the model on the test set.
    auc_score : float
        The AUC score of the model on the test set.
    aupr : float
        The AUPR score of the model on the test set.

    Notes
    -----
    This function makes predictions on the test set using the given model
    and computes several evaluation metrics. The results are printed to the
    console and returned as a tuple.
    """
    print('---------------- start testing ------------------------------')
    y_pred = test_pre(train_incidence_pos, incidence_test, model)
    score_t = torch.squeeze(y_pred)
    true_test_score = score_t.cpu().numpy()[:, 1]
    b_score = [int(s >= 0.5) for s in true_test_score]
    b_score_t = torch.argmax(score_t, dim=-1)
    y_test = y_test.cpu()
    auc_score = roc_auc_score(y_test, true_test_score)
    pr = precision_score(y_test, b_score)
    re = recall_score(y_test, b_score)
    f1 = f1_score(y_test, b_score)
    aupr = average_precision_score(y_test, true_test_score)
    print(
        f'f1:{f1},pr:{pr},recall:{re},auc:{auc_score},aupr:{aupr}')
    return f1, pr, re, auc_score, aupr

def statistic(whole_incidence_matrix):

    """
    Compute and save the row-wise sum of selected columns in the incidence matrix.

    Parameters
    ----------
    whole_incidence_matrix : pd.DataFrame
        The incidence matrix with columns containing underscores and integers
        for grouping.

    Returns
    -------
    None

    Notes
    -----
    The CSV file 'row_counts_by_column_group.csv' contains the row-wise counts
    of 1s for each group of columns, with each group represented as a separate
    column in the output DataFrame.
    """

    output_df = pd.DataFrame()

    for i in range(1, 11):
        selected_columns = [
            col for col in whole_incidence_matrix.columns 
            if int(col.split('_')[2]) == i
        ]
        
        filtered_matrix = whole_incidence_matrix[selected_columns]
        
        row_counts = filtered_matrix.sum(axis=1)
        
        output_df[f'count_for_{i}'] = row_counts

    output_df.to_csv('./row_counts_by_column_group.csv', index=False)

    print("Statistical results saved to row_counts_by_column_group.csv")

def split_data(incidence_matrix, labels, seed, train_frac=0.6, valid_frac=0.2):

    """
    Split the given incidence matrix and labels into training, validation, and testing sets.

    Parameters
    ----------
    incidence_matrix : pd.DataFrame
        The incidence matrix to split.
    labels : pd.Series
        The labels to split.
    seed : int
        The random seed for shuffling the data.
    train_frac : float, optional
        The proportion of samples to include in the training set (default is 0.6).
    valid_frac : float, optional
        The proportion of samples to include in the validation set (default is 0.2).

    Returns
    -------
    train_df : pd.DataFrame
        The training set of the incidence matrix.
    valid_df : pd.DataFrame
        The validation set of the incidence matrix.
    test_df : pd.DataFrame
        The testing set of the incidence matrix.
    train_labels : pd.Series
        The training set of labels.
    valid_labels : pd.Series
        The validation set of labels.
    test_labels : pd.Series
        The testing set of labels.

    Notes
    -----
    The returned DataFrames and Series are shuffled according to the given seed.
    """
    
    shuffled_matrix = incidence_matrix.sample(frac=1, axis=1, random_state=seed)
    shuffled_labels = labels[shuffled_matrix.columns]

    train_end = int(train_frac * shuffled_matrix.shape[1])
    valid_end = train_end + int(valid_frac * shuffled_matrix.shape[1])

    train_df = shuffled_matrix.iloc[:, :train_end]
    valid_df = shuffled_matrix.iloc[:, train_end:valid_end]
    test_df = shuffled_matrix.iloc[:, valid_end:]

    train_labels = shuffled_labels[:train_end]
    valid_labels = shuffled_labels[train_end:valid_end]
    test_labels = shuffled_labels[valid_end:]

    return train_df, valid_df, test_df, train_labels, valid_labels, test_labels

if __name__ == '__main__':
    args = config.parse()
    args.algorithm = "smiles2vec"
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    set_random_seed(args.seed)
    print(f"the seed is {args.seed}")
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # 1. load data
    dataset_base_dir = r'dataset\evaluation_subset\subset_drug2-8_SE5-50'
    training_sub_ds_names = ['2015Q1', '2015Q2', '2015Q3', '2016Q4', '2017Q1', '2017Q2', '2017Q3', '2017Q4', '2018Q3', '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q2']
    validating_sub_ds_names = ['2014Q3', '2015Q4', '2016Q1', '2016Q3', '2021Q2', '2024Q1']
    testing_sub_ds_names = ['2014Q4', '2016Q2', '2018Q1', '2018Q2', '2018Q4', '2024Q3']

    # 1.1 load smiles
    smiles_ds = pd.read_csv(r'dataset\dictionary\Drugbank_ID_SMILE_all_structure links.csv')
    drugbank_to_smiles = smiles_ds.loc[:, ['DrugBank ID', 'SMILES']].set_index('DrugBank ID').to_dict()['SMILES']

    all_ds_pos = pd.read_csv(r'dataset\evaluation_subset\subset_drug2-8_SE5-50\merged_subset\positive_samples_2014Q3_2024Q3_step6.csv')
    all_ds_neg = pd.read_csv(r'dataset\evaluation_subset\subset_drug2-8_SE5-50\merged_subset\negative_samples_2014Q3_2024Q3_step6.csv')
    all_ds = pd.concat([all_ds_pos, all_ds_neg], axis=0)

    drugs = set()
    for drug_ids in all_ds['DrugBankID']:
        drugs.update(ast.literal_eval(drug_ids))
    # Construct a mapping from drugs to indices
    drug_to_index = {drug: idx for idx, drug in enumerate(drugs)}

    # 2. merge datasets
    def merge_sub_datasets(sub_datasets):
        pos_merged_data = pd.DataFrame()
        neg_merged_data = pd.DataFrame()
        for sub_ds in sub_datasets:
            pos_file_path = f'{dataset_base_dir}{sub_ds}_positive_samples_condition123_SE_above_0.9.csv'
            neg_file_path = f'{dataset_base_dir}{sub_ds}_negative_samples_condition123_SE_above_0.9.csv'
            pos_sub_data = pd.read_csv(pos_file_path)
            pos_merged_data = pd.concat([pos_merged_data, pos_sub_data], axis=0)
            neg_sub_data = pd.read_csv(neg_file_path)
            neg_merged_data = pd.concat([neg_merged_data, neg_sub_data], axis=0)
        return pos_merged_data, neg_merged_data

    def build_incidence_matrix(pos_merged_data, neg_merged_data, drugs, drugbank_to_smiles):
        num_drugs = len(drugs)
        num_se_pos = len(pos_merged_data)
        num_se_neg = len(neg_merged_data)
        incidence_matrix_pos = np.zeros((num_drugs, num_se_pos), dtype=int)
        incidence_matrix_neg = np.zeros((num_drugs, num_se_neg), dtype=int)

        drug_id_list = [None] * len(drugs)
        for id in drug_to_index:
            drug_id_list[drug_to_index[id]] = id
        # Fill incidence matrix
        for col_idx, drug_list in tqdm(enumerate(pos_merged_data['DrugBankID']), total=len(pos_merged_data), desc='Fill incidence matrix for positive samples'):
            drug_ids = ast.literal_eval(drug_list)
            for drug_id in drug_ids:
                row_idx = drug_to_index[drug_id]
                if drug_id.lower() != 'none': 
                    incidence_matrix_pos[row_idx, col_idx] = 1

        for col_idx, drug_list in tqdm(enumerate(neg_merged_data['DrugBankID']), total=len(neg_merged_data), desc='Fill incidence matrix for negative samples'):
            drug_ids = ast.literal_eval(drug_list)
            for drug_id in drug_ids:
                row_idx = drug_to_index[drug_id]
                if drug_id.lower() != 'none':  
                    incidence_matrix_neg[row_idx, col_idx] = 1

        labels_pos = np.ones(incidence_matrix_pos.shape[1], dtype=int)
        labels_neg = np.zeros(incidence_matrix_neg.shape[1], dtype=int)

        incidence_matrix_pos = pd.DataFrame(incidence_matrix_pos)
        incidence_matrix_neg = pd.DataFrame(incidence_matrix_neg)

        incidence_matrix_pos.index = drug_id_list
        incidence_matrix_neg.index = drug_id_list

        old_row_count = len(incidence_matrix_pos)
        incidence_matrix_pos = incidence_matrix_pos.loc[incidence_matrix_pos.index.intersection(drugbank_to_smiles.keys())]
        incidence_matrix_neg = incidence_matrix_neg.loc[incidence_matrix_neg.index.intersection(drugbank_to_smiles.keys())]
        new_row_count = len(incidence_matrix_pos)
        print(f"{old_row_count - new_row_count} rows were deleted due to missing DrugBankIDs ({old_row_count}, {new_row_count})")
                    

        return incidence_matrix_pos, incidence_matrix_neg, labels_pos, labels_neg

    # 2.1 merge sub datasets
    # Get positive and negative samples
    train_data_pos, train_data_neg = merge_sub_datasets(training_sub_ds_names)
    train_incidence_matrix_pos, train_incidence_matrix_neg, train_labels_pos, train_labels_neg = build_incidence_matrix(train_data_pos, train_data_neg, drugs, drugbank_to_smiles)
    val_data_pos, val_data_neg = merge_sub_datasets(validating_sub_ds_names)
    val_incidence_matrix_pos, val_incidence_matrix_neg, val_labels_pos, val_labels_neg = build_incidence_matrix(val_data_pos, val_data_neg, drugs, drugbank_to_smiles)
    test_data_pos, test_data_neg = merge_sub_datasets(testing_sub_ds_names)
    test_incidence_matrix_pos, test_incidence_matrix_neg, test_labels_pos, test_labels_neg = build_incidence_matrix(test_data_pos, test_data_neg, drugs, drugbank_to_smiles)

    incidence_matrix_pos_df = pd.concat([train_incidence_matrix_pos, val_incidence_matrix_pos, test_incidence_matrix_pos], axis=1)

    # Get SMILES for positive samples
    X_smiles = [drugbank_to_smiles[x] for x in train_incidence_matrix_pos.index]
    X_smiles = list(X_smiles)

    results = {
        "f1": [],
        "pr": [],
        "re": [],
        "auc_score": [],
        "aupr": []
    }
    for i in range(args.iteration):
        print(f'---------------- iteration {i} ------------------------------')
        print('---------------- load samples ------------------------------')

        # Concatenate positive and negative samples
        train_df = pd.concat([train_incidence_matrix_pos, train_incidence_matrix_neg], axis=1)
        valid_df = pd.concat([val_incidence_matrix_pos, val_incidence_matrix_neg], axis=1)
        test_df = pd.concat([test_incidence_matrix_pos, test_incidence_matrix_neg], axis=1)

        train_df.columns = range(train_df.shape[1])
        valid_df.columns = range(valid_df.shape[1])
        test_df.columns = range(test_df.shape[1])   

        # Concatenate positive and negative labels
        y_train = np.concatenate([train_labels_pos, train_labels_neg])
        y_valid = np.concatenate([val_labels_pos, val_labels_neg])
        y_test = np.concatenate([test_labels_pos, test_labels_neg])

        # 1. Get shuffled column indices
        shuffled_train_columns = train_df.sample(frac=1, axis=1, random_state=args.seed).columns
        shuffled_test_columns = test_df.sample(frac=1, axis=1, random_state=args.seed).columns
        shuffled_valid_columns = valid_df.sample(frac=1, axis=1, random_state=args.seed).columns

        # 2. Reorder train_df and ensure y_train is in the same order
        train_df = train_df[shuffled_train_columns]
        y_train = y_train[shuffled_train_columns.to_numpy()]

        test_df = test_df[shuffled_test_columns]
        y_test = y_test[shuffled_test_columns.to_numpy()]

        valid_df = valid_df[shuffled_valid_columns]
        y_valid = y_valid[shuffled_valid_columns.to_numpy()]

        train_incidence_pos = torch.tensor(train_incidence_matrix_pos.to_numpy(), dtype=torch.float).to(device)
        incidence_train = torch.tensor(train_df.to_numpy(), dtype=torch.float).to(device)
        incidence_test = torch.tensor(test_df.to_numpy(), dtype=torch.float).to(device)
        incidence_valid = torch.tensor(valid_df.to_numpy(), dtype=torch.float).to(device)
        incidence_matrix_pos = torch.tensor(incidence_matrix_pos_df.to_numpy(), dtype=torch.float).to(device)

        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        y_valid = torch.tensor(y_valid, dtype=torch.long).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        model = train(args, X_smiles, incidence_matrix_pos, train_incidence_pos, incidence_train, incidence_valid, y_train, y_valid, i)
        f1, pr, re, auc_score, aupr = predict(model, train_incidence_pos, incidence_test, y_test)
        torch.save({'model': model.state_dict()}, f'{args.output}{args.train}_model_{i}.pth')
        
        results["f1"].append(f1)
        results["pr"].append(pr)
        results["re"].append(re)
        results["auc_score"].append(auc_score)
        results["aupr"].append(aupr)

    print(f"----------------final f1 of {args.train} with {args.algorithm}------------------------")
    # Calculate mean and standard deviation for each metric
    stats = {key: {"mean": np.mean(values), "std": np.std(values)} for key, values in results.items()}

    # Output the final statistics
    print("\nFinal Statistics After 10 Iterations:")
    for key, value in stats.items():
        print(f"{key}: Mean = {value['mean']:.4f}, Std = {value['std']:.4f}")
