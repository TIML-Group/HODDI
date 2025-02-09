# Set the environment variable
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import other modules
import os
import ast
import numpy as np
import pandas as pd 
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import precision_score, f1_score, roc_auc_score, average_precision_score
from datetime import datetime

# Use CUDNN
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  

# Import utils
from utils.model_utils import *

# Set random seed
seed = 123456
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader,
              optimizer, criterion, scaler, num_epochs, checkpoint_path, metrics_df, csv_file):
    """
    Train the model using the given data loaders, optimizer, and loss function.

    Args:
        model: The model to be trained.
        train_pos_loader: The data loader for the positive training data.
        train_neg_loader: The data loader for the negative training data.
        val_pos_loader: The data loader for the positive validation data.
        val_neg_loader: The data loader for the negative validation data.
        optimizer: The optimizer to use for training.
        criterion: The loss function to use for training.
        scaler: The gradient scaler to use for training.
        num_epochs: The number of epochs to train for.
        checkpoint_path: The path to save the model checkpoint to.
        metrics_df: The DataFrame to store the training and validation metrics in.
        csv_file: The path to save the metrics to.

    Returns:
        The best validation AUC achieved during training.
    """
    best_val_auc = 0
    patience = 10
    counter = 0
    min_delta = 0.001
    accumulation_steps = 4

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        epoch_loss = 0
        train_preds, train_labels = [], []
        optimizer.zero_grad()
        
        # Training loop
        for i, ((pos_drugs, pos_se, pos_label), (neg_drugs, neg_se, neg_label)) in enumerate(zip(train_pos_loader, train_neg_loader)):
            batch_drugs = torch.cat([pos_drugs, neg_drugs]).to(device)
            batch_se = torch.cat([pos_se, neg_se]).to(device)
            batch_labels = torch.cat([pos_label, neg_label]).to(device)
            
            with autocast():
                outputs = model(batch_drugs, batch_se)
                loss = criterion(outputs, batch_labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())
            
            if i % 100 == 0:
                torch.cuda.empty_cache()
        

        # Validation
        val_preds, val_labels = evaluate(model, val_pos_loader, val_neg_loader)
        val_preds_array = np.array(val_preds)
        val_labels_array = np.array(val_labels)
        val_loss = criterion(
            torch.from_numpy(val_preds_array).to(device), 
            torch.from_numpy(val_labels_array).to(device)
        ).item()
        
        val_metrics = calculate_metrics(val_preds, val_labels, val_loss, len(val_pos_loader), 'Val')
        val_auc = val_metrics['Val_AUC']  
        
        # Early stopping
        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss,
            }, checkpoint_path)
        else:
            counter += 1
        
        
        # Calculate metrics on training data
        train_metrics = calculate_metrics(train_preds, train_labels, epoch_loss, len(train_pos_loader), 'Train')
        metrics_df = update_metrics_df(metrics_df, epoch, train_metrics, val_metrics, csv_file)
               
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Loss: {train_metrics["Train_Loss"]:.4f}, Train AUC: {train_metrics["Train_AUC"]:.4f}, Val AUC: {val_metrics["Val_AUC"]:.4f}')
        
        if counter >= patience:
            print("Early stopping triggered")
            break
    
    return best_val_auc

def main(seed=seed):
    """
    Train an MLP model on the drug-side effect dataset.

    Args:
        seed (int): The random seed to use for reproducibility.

    Returns:
        model (nn.Module): The trained model.
        test_metrics (dict): The final test metrics.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Create output directory
    os.makedirs('model/MLP/v1', exist_ok=True)
    csv_file = f"model/MLP/v1/MLP_v1_seed{seed}.csv"
    checkpoint_path = f"model/MLP/v1/best_model_seed{seed}.pt"
    
    # Load quarterly data
    train_quarters, val_quarters, test_quarters = get_quarters()
    train_pos = merge_quarterly_data(train_quarters, 'positive')
    train_neg = merge_quarterly_data(train_quarters, 'negative')
    val_pos = merge_quarterly_data(val_quarters, 'positive')
    val_neg = merge_quarterly_data(val_quarters, 'negative')
    test_pos = merge_quarterly_data(test_quarters, 'positive')
    test_neg = merge_quarterly_data(test_quarters, 'negative')
    
    # Create data loaders
    dataloaders = create_dataloaders(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg)
    train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader, test_pos_loader, test_neg_loader = dataloaders
    
    # Initialize model and training components
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    metrics_df = pd.DataFrame(columns=['Epoch', 
                                 'Train_Loss', 'Train_Precision', 'Train_F1', 'Train_AUC', 'Train_PRAUC',
                                 'Val_Loss', 'Val_Precision', 'Val_F1', 'Val_AUC', 'Val_PRAUC'])
    
    # Train model
    train_model(model, train_pos_loader, train_neg_loader, val_pos_loader, val_neg_loader,
                optimizer, criterion, scaler, num_epochs=500, checkpoint_path=checkpoint_path,
                metrics_df=metrics_df, csv_file=csv_file)
    
    # Evaluate model
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    test_metrics = evaluate_model(model, test_pos_loader, test_neg_loader)
    
    print("\nFinal Test Results:")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return model, test_metrics

if __name__ == "__main__":
    main()