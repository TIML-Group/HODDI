# Import libraries
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Set model and tokenizer
smiles2vec_model_name = "./PubChem10M_SMILES_BPE_450k"
smiles2vec_model = AutoModelForMaskedLM.from_pretrained(smiles2vec_model_name)
smiles2vec_tokenizer = AutoTokenizer.from_pretrained(smiles2vec_model_name)

class SMILESDataset(Dataset):
    """ A dataset class used for storing SMILES data """
    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]
        
class CollateFunction:
    """A wrapper class that allows us to pass additional parameters like tokenizer to the collate function"""
    def __init__(self, tokenizer, batch_size):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, batch):
        """Process batch data"""
        if isinstance(batch, np.ndarray):
            batch = batch.tolist()
        batch = [str(x) for x in batch]   # Ensure all elements are strings
        tokens = self.tokenizer(batch, return_tensors="pt", max_length=self.batch_size, padding='max_length', truncation=True)
        return tokens

def smiles_to_chemberta_vector_gpu(smiles, batch_size=32):
    """
    Convert SMILES strings to ChemBERTa vectors
    :param smiles: List of SMILES strings
    :param model: Pre-trained ChemBERTa model
    :param tokenizer: Tokenizer for the ChemBERTa model
    :param batch_size: Size of each batch
    :return: n x 768 dimensional ChemBERTa vectors
    """
    # Create SMILES dataset
    dataset = SMILESDataset(smiles)
    # Create DataLoader
    collate_fn = CollateFunction(tokenizer=smiles2vec_tokenizer, batch_size=batch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smiles2vec_model.to(device)

    chemberta_feat = []

    for batch in tqdm(data_loader):
        # Move data to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]
        chemberta_vector = hidden_states.mean(dim=1)
        chemberta_feat.append(chemberta_vector.cpu())  # Move data to CPU

    chemberta_feat = torch.cat(chemberta_feat, dim=0)
    return chemberta_feat

def smiles_to_chemberta_vector(smiles, batch_size=32):
    """
    Convert SMILES strings to ChemBERTa vectors
    :param smiles: List of SMILES strings
    :param model: Pre-trained ChemBERTa model
    :param tokenizer: Tokenizer for the ChemBERTa model
    :return: 768 dimensional ChemBERTa vectors
    """
    chemberta_feat = []

    for smiles_item in tqdm(smiles):
        # Convert SMILES string to model input
        inputs = smiles2vec_tokenizer(smiles_item, return_tensors="pt", max_length=batch_size, padding='max_length', truncation=True)

        # Get model output, output hidden states
        with torch.no_grad():
            outputs = smiles2vec_model(**inputs, output_hidden_states=True)

        # Get the last layer's hidden state
        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden state

        # Take the mean of the last layer's hidden state as the vector representation of the SMILES string
        chemberta_vector = hidden_states.mean(dim=1)  # Convert tensor to NumPy array
        chemberta_feat.append(chemberta_vector)

    chemberta_feat = torch.tensor(np.vstack(chemberta_feat), dtype=torch.float32)
    return chemberta_feat