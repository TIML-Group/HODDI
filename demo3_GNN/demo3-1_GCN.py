# Set the environment variable
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Import necessary libraries
import os
import ast
import random
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GraphConv
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, f1_score, average_precision_score
from torchinfo import summary
from datetime import datetime

# Import necessary libraries for SMILES processing
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles

# Set the environment variable for CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
SMILES2VEC_MODEL = "seyonec/PubChem10M_SMILES_BPE_450k"
MAX_LENGTH = 512

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set the random seed
seed = 123456
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load SMILES data
def load_smiles_data():
    """
    Load SMILES data from CSV files.

    Returns:
        A dictionary where keys are drug bank IDs and values are their SMILES strings.
    """
    df1 = pd.read_csv(r'dataset\dictionary\Drugbank_ID_SMILE_all_structure links.csv')
    df2 = pd.read_csv(r'dataset\dictionary\DrugBankID2SMILES.csv')
    
    smiles_dict = {}
    for _, row in df1.iterrows():
        if pd.notna(row['SMILES']):
            smiles_dict[row['DrugBank ID']] = row['SMILES']
    for _, row in df2.iterrows():
        if pd.notna(row['smiles']):
            smiles_dict[row['drugbank_id']] = row['smiles']
    return smiles_dict


# Assistant classes
class SMILESDataset(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, idx):
        return self.smiles[idx]

class CollateFunction:
    def __init__(self, tokenizer, batch_size):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    def __call__(self, batch):
        return self.tokenizer(batch, return_tensors="pt", 
                            max_length=self.batch_size,
                            padding='max_length', 
                            truncation=True)


def prepare_data(pos_edges, neg_edges):
    """
    Prepare data for training a model.

    Parameters
    ----------
    pos_edges : pandas.DataFrame
        A DataFrame containing positive edges, with columns 'source', 'target', and 'SE_label'.
    neg_edges : pandas.DataFrame
        A DataFrame containing negative edges, with columns 'source', 'target', and 'SE_label'.
    
    Returns
    -------
    drug_to_idx : dict
        A dictionary mapping drug bank IDs to their indices.
    se_to_idx : dict
        A dictionary mapping side effect labels to their indices.
    pos_edges : pandas.DataFrame
        The input positive edges DataFrame.
    neg_edges : pandas.DataFrame
        The input negative edges DataFrame.
    smiles_dict : dict
        A dictionary mapping drug bank IDs to their SMILES strings.
    """
    # Get all unique drugs and side effects
    all_drugs = set(pos_edges['source']).union(set(pos_edges['target'])).union(set(neg_edges['source'])).union(set(neg_edges['target']))
    all_se = set(pos_edges['SE_label']).union(set(neg_edges['SE_label']))
    
    # Establish mappings for drugs and side effects
    drug_to_idx = {drug: idx for idx, drug in enumerate(all_drugs)}
    se_to_idx = {se: idx for idx, se in enumerate(all_se)}
    
    # Load SMILES data
    smiles_dict = load_smiles_data()
    
    return drug_to_idx, se_to_idx, pos_edges, neg_edges, smiles_dict

# Generate drug features
def get_drug_features(smiles_dict, drug_to_idx):
    
    """
    Generate feature vectors for drugs based on their SMILES representations.

    Parameters
    ----------
    smiles_dict : dict
        A dictionary mapping drug bank IDs to their SMILES strings.
    drug_to_idx : dict
        A dictionary mapping drug bank IDs to their indices.

    Returns
    -------
    torch.Tensor
        A tensor containing feature vectors for each drug. The feature vector
        dimension is assumed to be 768. Drugs without a SMILES representation
        are assigned zero vectors.
    """

    all_drugs = list(drug_to_idx.keys())
    feature_vectors = []
    vector_dim = 768  
    
    smiles_list = []
    drug_order = []
    
    for drug in all_drugs:
        if drug in smiles_dict:
            smiles_list.append(smiles_dict[drug])
            drug_order.append(drug)
        else:
            # Use zero vector to represent missing SMILES
            feature_vectors.append(torch.zeros(vector_dim))
    
    if smiles_list:
        vectors = smiles_to_vec(smiles_list)
        for vec in vectors:
            feature_vectors.append(vec)
    
    return torch.stack(feature_vectors)

def get_se_features(se_to_idx):
    
    """
    Generate feature vectors for side effects based on their descriptions.

    Parameters
    ----------
    se_to_idx : dict
        A dictionary mapping side effects to their indices.

    Returns
    -------
    torch.Tensor
        A tensor containing feature vectors for each side effect. The feature vector
        dimension is assumed to be 768. Side effects without a description are assigned
        zero vectors.
    """
    # Load side effect descriptions
    se_data = pd.read_csv('dataset/dictionary/SE_similarity_2014Q3_2024Q3.csv')
    se_descriptions = [se_data['recommended_umls_cui_from_meddra'][i] 
                      for i in range(len(se_to_idx))]
    
    # Generate feature vectors
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    
    features = []
    batch_size = 32
    for i in range(0, len(se_descriptions), batch_size):
        batch = se_descriptions[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            batch_features = outputs.hidden_states[-1][:, 0, :]
            features.append(batch_features.cpu())
    
    return torch.cat(features, dim=0)

# Build graph
def build_hetero_graph(drug_to_idx, se_to_idx, pos_edges, neg_edges, smiles_dict):
    
    """
    Construct a heterogeneous graph from the given data.

    Parameters
    ----------
    drug_to_idx : dict
        A dictionary mapping drug IDs to their indices.
    se_to_idx : dict
        A dictionary mapping side effect IDs to their indices.
    pos_edges : pd.DataFrame
        A DataFrame containing the positive edges, with columns 'source', 'target', and 'SE_label'.
    neg_edges : pd.DataFrame
        A DataFrame containing the negative edges, with columns 'source', 'target', and 'SE_label'.
    smiles_dict : dict
        A dictionary mapping drug IDs to their SMILES strings.

    Returns
    -------
    data : torch_geometric.data.HeteroData
        The constructed heterogeneous graph.
    """
    
    # Initialize edge index and label dictionaries
    edge_index_dict = {
        ('drug', 'interacts', 'drug'): [],
        ('drug', 'causes', 'side_effect'): [],
        ('side_effect', 'caused_by', 'drug'): []
    }
    
    edge_label_dict = {
        ('drug', 'interacts', 'drug'): [],
        ('drug', 'causes', 'side_effect'): [],
        ('side_effect', 'caused_by', 'drug'): []
    }
    
    # Add positive edges and labels
    for _, row in pos_edges.iterrows():
        source = drug_to_idx[row['source']]
        target = drug_to_idx[row['target']]
        se = se_to_idx[row['SE_label']]
        
        # Add drug-drug edges
        edge_index_dict[('drug', 'interacts', 'drug')].append([source, target])
        edge_label_dict[('drug', 'interacts', 'drug')].append(1)
        
        # Add drug-side effect edges
        edge_index_dict[('drug', 'causes', 'side_effect')].append([source, se])
        edge_label_dict[('drug', 'causes', 'side_effect')].append(1)
        
        # Add side effect-drug edges
        edge_index_dict[('side_effect', 'caused_by', 'drug')].append([se, source])
        edge_label_dict[('side_effect', 'caused_by', 'drug')].append(1)
    
    # Add negative edges and labels
    for _, row in neg_edges.iterrows():
        source = drug_to_idx[row['source']]
        target = drug_to_idx[row['target']]
        se = se_to_idx[row['SE_label']]
        
        # Add drug-drug edges
        edge_index_dict[('drug', 'interacts', 'drug')].append([source, target])
        edge_label_dict[('drug', 'interacts', 'drug')].append(0)
        
        # Add drug-side effect edges
        edge_index_dict[('drug', 'causes', 'side_effect')].append([source, se])
        edge_label_dict[('drug', 'causes', 'side_effect')].append(0)
        
        # Add side effect-drug edges
        edge_index_dict[('side_effect', 'caused_by', 'drug')].append([se, source])
        edge_label_dict[('side_effect', 'caused_by', 'drug')].append(0)
    
    # Convert edge index and label lists to tensors
    for key in edge_index_dict:
        if edge_index_dict[key]:
            edge_index_dict[key] = torch.tensor(edge_index_dict[key]).t().contiguous()
            edge_label_dict[key] = torch.tensor(edge_label_dict[key])
        else:
            edge_index_dict[key] = torch.empty((2, 0), dtype=torch.long)
            edge_label_dict[key] = torch.empty(0, dtype=torch.long)
    
    # Construct features
    drug_features = get_drug_features(smiles_dict, drug_to_idx)
    se_features = get_se_features(se_to_idx)
    
    data = HeteroData()
    data['drug'].x = drug_features
    data['side_effect'].x = se_features
    
    # Add edges and labels
    for key in edge_index_dict:
        data[key].edge_index = edge_index_dict[key]
        data[key].edge_label = edge_label_dict[key]
    

    # Add periods for positive and negative edges
    edge_periods = []
    # drug-drug edges
    edge_periods.extend(pos_edges['period'])  
    edge_periods.extend(neg_edges['period'])
    # drug-SE edges
    edge_periods.extend(pos_edges['period'])
    edge_periods.extend(neg_edges['period'])
    # SE-drug edges 
    edge_periods.extend(pos_edges['period'])
    edge_periods.extend(neg_edges['period'])
    
    data.periods = edge_periods
    
    return data

def add_edges(pos_edges, edge_index_dict, drug_to_idx, se_to_idx):
    
    """
    Add edges to the edge index dictionary based on positive edge data.

    Parameters
    ----------
    pos_edges : pandas.DataFrame
        A DataFrame containing positive edges, with columns 'DrugBankID' and 'SE_above_0.9'.
    edge_index_dict : dict
        A dictionary with keys as edge types and values as lists of edge indices.
    drug_to_idx : dict
        A dictionary mapping drug bank IDs to their indices.
    se_to_idx : dict
        A dictionary mapping side effect labels to their indices.
    """

    for _, row in pos_edges.iterrows():
       drugs = ast.literal_eval(row['DrugBankID'])
       se = row['SE_above_0.9']
       
       # drug-drug edges
       for i in range(len(drugs)):
           for j in range(i+1, len(drugs)):
               if drugs[i] in drug_to_idx and drugs[j] in drug_to_idx:
                   edge_index_dict[('drug', 'interacts', 'drug')].append(
                       [drug_to_idx[drugs[i]], drug_to_idx[drugs[j]]])
                   edge_index_dict[('drug', 'interacts', 'drug')].append(
                       [drug_to_idx[drugs[j]], drug_to_idx[drugs[i]]])
       
       # drug-SE edges
       if se in se_to_idx:
           for drug in drugs:
               if drug in drug_to_idx:
                   edge_index_dict[('drug', 'causes', 'side_effect')].append(
                       [drug_to_idx[drug], se_to_idx[se]])
                   edge_index_dict[('side_effect', 'caused_by', 'drug')].append(
                       [se_to_idx[se], drug_to_idx[drug]])

def add_negative_samples(edge_index_dict, drug_to_idx):
    
    """
    Add negative edges to the edge index dictionary. The negative edges are
    determined as drug pairs with similarity < 0.3 based on Morgan fingerprints.
    
    Parameters
    ----------
    edge_index_dict : dict
        A dictionary with keys as edge types and values as lists of edge indices.
    drug_to_idx : dict
        A dictionary mapping drug bank IDs to their indices.
    
    Returns
    -------
    None
    """
    smiles_dict = load_smiles_data()
    fps = {}
    for drug_id, smiles in smiles_dict.items():
        if drug_id in drug_to_idx:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fps[drug_id] = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    
    existing_edges = set(map(tuple, edge_index_dict[('drug', 'interacts', 'drug')]))
    negative_edges = []
    
    for drug1 in fps:
        for drug2 in fps:
            if drug1 != drug2:
                pair = (drug_to_idx[drug1], drug_to_idx[drug2])
                if pair not in existing_edges:
                    similarity = DataStructs.TanimotoSimilarity(fps[drug1], fps[drug2])
                    if similarity < 0.3:
                        negative_edges.append(list(pair))
    
    num_pos = len(edge_index_dict[('drug', 'interacts', 'drug')])
    negative_edges = negative_edges[:num_pos]
    edge_index_dict[('drug', 'interacts', 'drug')].extend(negative_edges)


def smiles_to_vec(smiles_list):
    """
    Convert a list of SMILES strings into feature vectors using a pre-trained model.

    Parameters
    ----------
    smiles_list : list of str
        A list of SMILES strings representing chemical compounds.

    Returns
    -------
    torch.Tensor
        A tensor containing the feature vectors for each SMILES string. The feature
        vector dimension is assumed to be 768.
    """

    tokenizer = AutoTokenizer.from_pretrained(SMILES2VEC_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(SMILES2VEC_MODEL)
    model = model.to(device)
    
    BATCH_SIZE = 16
    MAX_SEQ_LENGTH = 256
    
    dataset = SMILESDataset(smiles_list)
    collate_fn = CollateFunction(tokenizer=tokenizer, batch_size=MAX_SEQ_LENGTH)
    data_loader = DataLoader(dataset, 
                            batch_size=BATCH_SIZE, 
                            collate_fn=collate_fn,
                            shuffle=False)
    
    vectors = []
    model.eval()
    
    for batch in tqdm(data_loader, desc="Processing SMILES batches"):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                batch_vectors = hidden_states.mean(dim=1)
                vectors.append(batch_vectors.cpu())
                
        except RuntimeError as e:
            print(f"Error: {str(e)}")
            batch_size = len(batch['input_ids'])
            vectors.append(torch.zeros(batch_size, 768).cpu())
            continue
            
    vectors = torch.cat(vectors, dim=0)
    return vectors


# Model definition
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.feature_norm = torch.nn.BatchNorm1d(hidden_channels)
        self.conv1 = HeteroConv({
            ('drug', 'interacts', 'drug'): GraphConv(-1, hidden_channels),
            ('drug', 'causes', 'side_effect'): GraphConv(-1, hidden_channels),
            ('side_effect', 'caused_by', 'drug'): GraphConv(-1, hidden_channels)
        })
        self.conv2 = HeteroConv({
            ('drug', 'interacts', 'drug'): GraphConv(hidden_channels, hidden_channels),
            ('drug', 'causes', 'side_effect'): GraphConv(hidden_channels, hidden_channels),
            ('side_effect', 'caused_by', 'drug'): GraphConv(hidden_channels, hidden_channels)
        })
        self.classifier = torch.nn.Linear(hidden_channels*2, 2)

    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        x_dict1 = {key: F.leaky_relu(x) for key, x in x_dict1.items()}
        
        x_dict2 = self.conv2(x_dict1, edge_index_dict)
        x_dict2 = {key: F.leaky_relu(x) for key, x in x_dict2.items()}
        
        x_dict = {key: self.feature_norm(x) for key, x in x_dict2.items()}
        return x_dict
    
    def predict(self, x_dict, edge_index_dict, edge_type):
        src, dst = edge_index_dict[edge_type]
        src_embedding = x_dict[edge_type[0]][src]
        dst_embedding = x_dict[edge_type[2]][dst]
        edge_embedding = torch.cat([src_embedding, dst_embedding], dim=1)
        return self.classifier(edge_embedding)


def train_model(data, model, device, train_periods, val_periods, test_periods,seed=seed):
    """
    Train the HeteroGNN model with the given data and hyperparameters.

    Args:
    - data: The dataset to train the model on.
    - model: The HeteroGNN model to train.
    - device: The device to use for training.
    - train_periods: The periods to use for training.
    - val_periods: The periods to use for validation.
    - test_periods: The periods to use for testing.
    - seed: The seed to use for reproducibility.

    Returns:
    - model: The trained model.
    - metrics: A dictionary containing the final test metrics.
    """
    save_dir = "model/GNN/hetero_GraphConv/vanilla/"
    os.makedirs(save_dir, exist_ok=True)

    # Get edge indices and labels
    num_edges_dd = data[('drug', 'interacts', 'drug')].edge_index.size(1)
    num_edges_dse = data[('drug', 'causes', 'side_effect')].edge_index.size(1)

    # Create period masks
    train_mask = torch.tensor([period in train_periods for period in data.periods])
    val_mask = torch.tensor([period in val_periods for period in data.periods])
    test_mask = torch.tensor([period in test_periods for period in data.periods])

    # Split edge indices into drug-drug and drug-SE
    train_idx_dd = torch.where(train_mask[:num_edges_dd])[0]
    train_idx_dse = torch.where(train_mask[num_edges_dd:])[0]
    val_idx_dd = torch.where(val_mask[:num_edges_dd])[0]
    val_idx_dse = torch.where(val_mask[num_edges_dd:])[0]
    test_idx_dd = torch.where(test_mask[:num_edges_dd])[0]
    test_idx_dse = torch.where(test_mask[num_edges_dd:])[0]

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Get labels
    drug_drug_labels = data[('drug', 'interacts', 'drug')].edge_label.long()
    drug_se_labels = data[('drug', 'causes', 'side_effect')].edge_label.long()
    
    # Check index range
    train_idx_dd = train_idx_dd[train_idx_dd < len(drug_drug_labels)]
    train_idx_dse = train_idx_dse[train_idx_dse < len(drug_se_labels)]
    val_idx_dd = val_idx_dd[val_idx_dd < len(drug_drug_labels)]
    val_idx_dse = val_idx_dse[val_idx_dse < len(drug_se_labels)]
    test_idx_dd = test_idx_dd[test_idx_dd < len(drug_drug_labels)]
    test_idx_dse = test_idx_dse[test_idx_dse < len(drug_se_labels)]
    
    
    if len(train_idx_dd) > 0 and len(train_idx_dse) > 0:
        pos_count_dd = (drug_drug_labels[train_idx_dd] == 1).sum().float()
        neg_count_dd = (drug_drug_labels[train_idx_dd] == 0).sum().float()
        pos_weight_dd = torch.clamp(neg_count_dd / (pos_count_dd + 1e-6), max=10.0)
        
        pos_count_dse = (drug_se_labels[train_idx_dse] == 1).sum().float()
        neg_count_dse = (drug_se_labels[train_idx_dse] == 0).sum().float()
        pos_weight_dse = torch.clamp(neg_count_dse / (pos_count_dse + 1e-6), max=10.0)
    else:
        pos_weight_dd = torch.tensor(1.0).to(device)
        pos_weight_dse = torch.tensor(1.0).to(device)
    
    # Define loss function with class weights
    criterion_dd = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight_dd.item()]).to(device))
    criterion_dse = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight_dse.item()]).to(device))
    
    # Initialize best metrics
    best_val_auc = 0
    patience = 20
    counter = 0
    min_delta = 0.001
    alpha = 1 # Balance coeffient between drug-drug and drug-side effect relationships
    
    # Prepare CSV file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(save_dir, f"GraphConv_vanilla_seed{seed}.csv")
    metrics_df = pd.DataFrame(columns=['Epoch', 'Loss', 'Val_AUC_DD', 'Val_AUC_DSE',
                                     'Precision', 'F1_Score', 'AUC', 'AUPRC'])
    
    # Training loop
    for epoch in tqdm(range(500), desc="Training"):
        model.train()
        optimizer.zero_grad()
        
        x_dict = model(data.x_dict, data.edge_index_dict)
        
        # Calculate loss for drug-drug
        outputs_dd = model.predict(x_dict, data.edge_index_dict, ('drug', 'interacts', 'drug'))
        loss_dd = criterion_dd(
            outputs_dd[train_idx_dd],
            drug_drug_labels[train_idx_dd]
        )
        
        # Calculate loss for drug-side_effect
        outputs_dse = model.predict(x_dict, data.edge_index_dict, ('drug', 'causes', 'side_effect'))
        loss_dse = criterion_dse(
            outputs_dse[train_idx_dse],
            drug_se_labels[train_idx_dse]
        )
        
        
        loss = (1-alpha)*loss_dd + alpha*loss_dse
        loss.backward()
        optimizer.step()
        
        # Model evaluation
        model.eval()
        with torch.no_grad():
            # Validation for drug-drug
            val_outputs_dd = model.predict(x_dict, data.edge_index_dict, ('drug', 'interacts', 'drug'))
            val_predictions_dd = F.softmax(val_outputs_dd[val_idx_dd], dim=1)[:, 1]
            val_auc_dd = roc_auc_score(
                drug_drug_labels[val_idx_dd].cpu(),
                val_predictions_dd.cpu()
            )
            
            # Validation for drug-side_effect
            val_outputs_dse = model.predict(x_dict, data.edge_index_dict, ('drug', 'causes', 'side_effect'))
            val_predictions_dse = F.softmax(val_outputs_dse[val_idx_dse], dim=1)[:, 1]
            val_auc_dse = roc_auc_score(
                drug_se_labels[val_idx_dse].cpu(),
                val_predictions_dse.cpu()
            )
            
            # Validation AUC for both drug-drug and drug-side_effect
            val_auc = ((1-alpha)*val_auc_dd + alpha*val_auc_dse) / 2
            scheduler.step(val_auc)
            
            # Save the best model
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f'best_model_seed{seed}.pt')
            
            if val_auc > best_val_auc + min_delta:
                best_val_auc = val_auc
                counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'epoch': epoch,
                    'loss': loss,
                }, checkpoint_path)
            else:
                counter += 1
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Test 
            with torch.no_grad():
                # Test for drug-drug
                test_outputs_dd = model.predict(x_dict, data.edge_index_dict, ('drug', 'interacts', 'drug'))
                test_predictions_dd = F.softmax(test_outputs_dd[test_idx_dd], dim=1)[:, 1]
                test_auc_dd = roc_auc_score(
                    drug_drug_labels[test_idx_dd].cpu(),
                    test_predictions_dd.cpu()
                )
                
                # Test for drug-side_effect
                test_outputs_dse = model.predict(x_dict, data.edge_index_dict, ('drug', 'causes', 'side_effect'))
                test_predictions_dse = F.softmax(test_outputs_dse[test_idx_dse], dim=1)[:, 1]
                test_auc_dse = roc_auc_score(
                    drug_se_labels[test_idx_dse].cpu(),
                    test_predictions_dse.cpu()
                )
                
                # Comprehensive test evaluation
                test_predictions =test_predictions_dse.cpu()
                test_true = drug_se_labels[test_idx_dse].cpu()
                test_precision = precision_score(test_true, (test_predictions > 0.5).int(), zero_division=0)
                test_f1 = f1_score(test_true, (test_predictions > 0.5).int(), zero_division=0)
                test_auc = roc_auc_score(test_true, test_predictions)
                test_auprc = average_precision_score(test_true, test_predictions)
        
            temp_df = pd.DataFrame({
                'Epoch': [epoch],
                'Loss': [loss.item()],
                'Val_AUC_DD': [val_auc_dd],
                'Val_AUC_DSE': [val_auc_dse],
                'Precision': [test_precision],
                'F1_Score': [test_f1],
                'AUC': [test_auc],
                'AUPRC': [test_auprc]
            })
            metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)
            
            if epoch % 10 == 0:
                metrics_df.to_csv(csv_file, index=False)
    
    metrics_df.to_csv(csv_file, index=False)
    
    # Load the best model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Final test metrics
    with torch.no_grad():
        # Predict metrics on the test set for drug-drug
        test_outputs_dd = model.predict(x_dict, data.edge_index_dict, ('drug', 'interacts', 'drug'))
        test_predictions_dd = F.softmax(test_outputs_dd[test_idx_dd], dim=1)[:, 1]
        test_auc_dd = roc_auc_score(
            drug_drug_labels[test_idx_dd].cpu(),
            test_predictions_dd.cpu()
        )
        
        # Predict metrics on the test set for drug-side_effect
        test_outputs_dse = model.predict(x_dict, data.edge_index_dict, ('drug', 'causes', 'side_effect'))
        test_predictions_dse = F.softmax(test_outputs_dse[test_idx_dse], dim=1)[:, 1]
        test_auc_dse = roc_auc_score(
            drug_se_labels[test_idx_dse].cpu(),
            test_predictions_dse.cpu()
        )
        
        # Evaluate on the test set
        test_predictions = test_predictions_dse.cpu()
        test_true = drug_se_labels[test_idx_dse].cpu()
        test_precision = precision_score(test_true, (test_predictions > 0.5).int(), zero_division=0)
        test_f1 = f1_score(test_true, (test_predictions > 0.5).int(), zero_division=0)
        test_auc = roc_auc_score(test_true, test_predictions)
        test_auprc = average_precision_score(test_true, test_predictions)
    
    # Final test metrics
    metrics = {
        'precision': test_precision,
        'f1': test_f1,
        'auc': test_auc,
        'auprc': test_auprc,
        'auc_DD': test_auc_dd,
        'auc_DSE': test_auc_dse
    }
    
    print("\nFinal Test Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model, metrics


 
def main():
    
    """
    Train a HeteroGNN model with GraphConv on the given data and hyperparameters.

    Parameters
    ----------
    None

    Returns
    -------
    model : torch.nn.Module
        The trained model.
    metrics : dict
        A dictionary containing the final test metrics.
    """
    
    # Save directory
    save_dir = "model/GNN/hetero_GraphConv/vanilla/"
    os.makedirs(save_dir, exist_ok=True)

    # Data splitting
    train_periods = ['2015Q1', '2015Q4', '2016Q1', '2016Q3', '2016Q4', '2017Q1', '2017Q3', '2017Q4', '2018Q3', '2019Q2', '2019Q4', '2020Q1', '2020Q2', '2020Q3', '2020Q4', '2021Q1', '2021Q2', '2021Q3', '2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1', '2024Q3']
    val_periods = ['2016Q2', '2018Q1', '2018Q2', '2018Q4', '2019Q3', '2024Q2']
    test_periods = ['2014Q3', '2014Q4', '2015Q2', '2015Q3', '2017Q2', '2019Q1']

    base_dir = "dataset/converted_evaluation_subset/subset_drug2-8_SE5-50/clique_expansion/"

    # Collect positive and negative samples
    pos_edges = []
    neg_edges = []

    # Process training, validation, and test sets
    for period in train_periods + val_periods + test_periods:
        pos_file = f"{base_dir}/{period}_positive_samples_condition123_SE_above_0.9_clique_expansion.csv"
        neg_file = f"{base_dir}/{period}_negative_samples_condition123_SE_above_0.9_clique_expansion.csv"
        
        pos_df = pd.read_csv(pos_file)
        neg_df = pd.read_csv(neg_file)
        
        # Add time tag
        pos_df['period'] = period
        neg_df['period'] = period
        
        pos_edges.append(pos_df)
        neg_edges.append(neg_df)

    pos_edges = pd.concat(pos_edges, ignore_index=True)
    neg_edges = pd.concat(neg_edges, ignore_index=True)
   
    
    # Prepare data
    drug_to_idx, se_to_idx, pos_edges, neg_edges, smiles_dict = prepare_data(pos_edges, neg_edges)
    
    # Construct heterogeneous graph
    data = build_hetero_graph(drug_to_idx, se_to_idx, pos_edges, neg_edges, smiles_dict).to(device)
    
    # Train model
    model = HeteroGNN(hidden_channels=128).to(device)
    model, metrics = train_model(data, model, device, train_periods, val_periods, test_periods)
    
    logger.info("Final test results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    # Train, validate, and test the model
    main() 
    



