# Set the environment variable
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Add parent directory to path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import necessary libraries
import os
import ast
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import precision_score, f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset

# Set cudnn benchmark
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # Use cudnn autotuner

# Set random seed
seed = 123456
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_dim=768):
        """
        Constructor for MLP model.

        Args:
            input_dim (int): Dimension of input features. Defaults to 768.
        """
        super().__init__()
        self.drug_reduction = nn.Linear(input_dim, 64)
        self.se_reduction = nn.Linear(input_dim, 64)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, drug_features, se_features):
        """
        Forward pass for the MLP model.

        Args:
            drug_features (torch.Tensor): Features for drugs.
            se_features (torch.Tensor): Features for side effects.

        Returns:
            torch.Tensor: Predictions for the input features.
        """
        drug_reduced = self.drug_reduction(drug_features)
        se_reduced = self.se_reduction(se_features)
        combined = torch.cat([drug_reduced, se_reduced], dim=1)
        return self.classifier(combined)


class ProcessedDataset(Dataset):
    def __init__(self, data):
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        feature_cols = [col for col in self.data.columns if col.startswith('feature_')]
        self.features = self.data[feature_cols].values
        self.labels = (self.data['hyperedge_label'] == 1).astype(float).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        drug_features = features[:768]  # The first 768 dimensions are drug features
        se_features = features[768:]    # The last 768 dimensions are SE features
        return (torch.FloatTensor(drug_features),
                torch.FloatTensor(se_features),
                torch.FloatTensor([self.labels[idx]]))

def get_quarters():
    """
    Returns train, validation, and test quarters.

    Returns:
        tuple: 3-element tuple containing the list of train quarters, validation quarters, and test quarters
    """
    train_quarters = ['2015Q1', '2015Q4', '2016Q1', '2016Q3', '2016Q4', '2017Q1', '2017Q3', '2017Q4', '2018Q3', '2019Q2', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1', '2024Q3']
    val_quarters = ['2016Q2', '2018Q1', '2018Q2', '2018Q4', '2019Q3', '2024Q2']
    test_quarters = ['2014Q3', '2014Q4', '2015Q2', '2015Q3', '2017Q2', '2019Q1']
    return train_quarters, val_quarters, test_quarters

def merge_quarterly_data(quarters, type_suffix):
    """
    Merge quarterly data from CSV files into a single DataFrame.

    Parameters:
        quarters (list): List of quarter strings (e.g., '2014Q3')
        type_suffix (str): Suffix to append to quarter name (e.g., 'positive')

    Returns:
        pd.DataFrame: Concatenated DataFrame
    """
    dfs = []
    base_path = 'dataset/MLP_input'
    for q in quarters:
        file_path = f"{base_path}/{q}_{type_suffix}_embeddings.csv"
        if os.path.exists(file_path):
            dfs.append(pd.read_csv(file_path))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def create_dataloaders(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg):
    """
    Create DataLoaders for training, validation, and testing datasets.

    Parameters:
        train_pos (pd.DataFrame): Positive samples for training
        train_neg (pd.DataFrame): Negative samples for training
        val_pos (pd.DataFrame): Positive samples for validation
        val_neg (pd.DataFrame): Negative samples for validation
        test_pos (pd.DataFrame): Positive samples for testing
        test_neg (pd.DataFrame): Negative samples for testing

    Returns:
        tuple[DataLoader]: (train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader, test_pos_loader, test_neg_loader)
    """
    loader_params = {
        'batch_size': 16,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True
    }
    
    return (
        DataLoader(ProcessedDataset(train_pos), shuffle=True, **loader_params),
        DataLoader(ProcessedDataset(train_neg), shuffle=True, **loader_params),
        DataLoader(ProcessedDataset(val_pos), **loader_params),
        DataLoader(ProcessedDataset(val_neg), **loader_params),
        DataLoader(ProcessedDataset(test_pos), **loader_params),
        DataLoader(ProcessedDataset(test_neg), **loader_params)
    )


def calculate_metrics(preds, labels, epoch_loss, num_batches, prefix='Train'):
    """
    Calculate evaluation metrics for model predictions.

    Parameters:
        preds (list or np.ndarray): Predicted probabilities for the positive class.
        labels (list or np.ndarray): True binary labels.
        epoch_loss (float): Total loss for the epoch.
        num_batches (int): Number of batches in the epoch.
        prefix (str, optional): Prefix for metric names, indicating the dataset split (e.g., 'Train', 'Val'). Default is 'Train'.

    Returns:
        dict: Dictionary containing the calculated metrics:
            - '{prefix}_Loss': Average loss per batch.
            - '{prefix}_Precision': Precision score.
            - '{prefix}_F1': F1 score.
            - '{prefix}_AUC': ROC AUC score.
            - '{prefix}_PRAUC': Precision-Recall AUC score.
    """

    predictions = (np.array(preds) > 0.5).astype(int)
    return {
        f'{prefix}_Loss': epoch_loss / num_batches,
        f'{prefix}_Precision': precision_score(labels, predictions, zero_division=0),
        f'{prefix}_F1': f1_score(labels, predictions, zero_division=0),
        f'{prefix}_AUC': roc_auc_score(labels, preds),
        f'{prefix}_PRAUC': average_precision_score(labels, preds)
    }



def update_metrics_df(metrics_df, epoch, train_metrics, val_metrics, csv_file):
    """
    Update the metrics DataFrame with new training and validation metrics for the current epoch.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing the metrics for all epochs.
        epoch (int): The current epoch number.
        train_metrics (dict): Dictionary of training metrics for the current epoch.
        val_metrics (dict): Dictionary of validation metrics for the current epoch.
        csv_file (str): Path to the CSV file where the updated metrics DataFrame will be saved.

    Returns:
        pd.DataFrame: The updated metrics DataFrame including the new row for the current epoch.
    """

    new_row = pd.DataFrame({
        'Epoch': [epoch + 1],
        **train_metrics,  
        **val_metrics    
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    metrics_df.to_csv(csv_file, index=False)
    return metrics_df


def evaluate(model, pos_loader, neg_loader):
    """
    Evaluate the model on the given positive and negative data loaders.

    Parameters:
        model (nn.Module): The model to evaluate.
        pos_loader (DataLoader): DataLoader for the positive samples.
        neg_loader (DataLoader): DataLoader for the negative samples.

    Returns:
        tuple[list, list]: A tuple of two lists. The first list contains the model's predictions for all samples.
        The second list contains the corresponding labels for all samples.
    """

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (pos_drugs, pos_se, pos_label), (neg_drugs, neg_se, neg_label) in zip(pos_loader, neg_loader):
            batch_drugs = torch.cat([pos_drugs, neg_drugs]).to(device)
            batch_se = torch.cat([pos_se, neg_se]).to(device)
            batch_labels = torch.cat([pos_label, neg_label]).to(device)
            
            with autocast():
                outputs = model(batch_drugs, batch_se)
            
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    return all_preds, all_labels


def evaluate_model(model, test_pos_loader, test_neg_loader):
    """
    Evaluate the model on the given positive and negative test data loaders.

    Parameters:
        model (nn.Module): The model to evaluate.
        test_pos_loader (DataLoader): DataLoader for the positive test samples.
        test_neg_loader (DataLoader): DataLoader for the negative test samples.

    Returns:
        dict: A dictionary containing the evaluation metrics for the model on the test data:
            - 'precision': Precision score.
            - 'f1': F1 score.
            - 'auc': ROC AUC score.
            - 'prauc': Precision-Recall AUC score.
    """

    test_preds, test_labels = evaluate(model, test_pos_loader, test_neg_loader)
    test_predictions = (np.array(test_preds) > 0.5).astype(int)
    
    return {
        'precision': precision_score(test_labels, test_predictions, zero_division=0),
        'f1': f1_score(test_labels, test_predictions, zero_division=0),
        'auc': roc_auc_score(test_labels, test_preds),
        'prauc': average_precision_score(test_labels, test_preds)
    }