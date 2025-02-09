# Set the environment variable and device
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import libraries
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_smiles_data():
    
    """
    Load SMILES data from CSV files.

    Returns:
        A dictionary where keys are drug bank IDs and values are their SMILES strings.
    """
    
    df1 = pd.read_csv('dataset/dictionary/Drugbank_ID_SMILE_all_structure links.csv')
    df2 = pd.read_csv('dataset/dictionary/DrugBankID2SMILES.csv')
    
    smiles_dict = {}
    for _, row in df1.iterrows():
        if pd.notna(row['SMILES']):
            smiles_dict[row['DrugBank ID']] = row['SMILES']
    for _, row in df2.iterrows():
        if pd.notna(row['smiles']):
            smiles_dict[row['drugbank_id']] = row['smiles']
    return smiles_dict

class EmbeddingProcessor:
    def __init__(self):
        """
        Initialize the EmbeddingProcessor object.

        This method loads the SMILES tokenzier and model, and the side effects data.

        """
        
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.bert_model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(device)
        self.bert_model.eval()
        self.side_effects = pd.read_csv('dataset/dictionary/Side_effects_unique.csv')
        self.smiles_dict = load_smiles_data()

    @torch.no_grad()
    def get_drugs_embedding(self, drug_ids):
        
        """
        Get the embedding of a list of drug bank IDs using a pre-trained SMILES BERT model.

        Parameters
        ----------
        drug_ids : str
            A string representation of a tuple of drug bank IDs.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of size 768 representing the embedding of the drug compounds.
        """
        drug_ids_tuple = tuple(ast.literal_eval(drug_ids))
        valid_smiles = [self.smiles_dict[drug_id] for drug_id in drug_ids_tuple 
                       if drug_id in self.smiles_dict and self.smiles_dict[drug_id]]
        
        if not valid_smiles:
            return np.zeros(768)
            
        try:
            inputs = self.tokenizer(
                valid_smiles,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            with autocast():
                outputs = self.bert_model(**inputs, output_hidden_states=True)
                embedding = outputs.hidden_states[-1].mean(dim=1).mean(dim=0).cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"Error processing drug {drug_ids}: {e}")
            return np.zeros(768)

    def get_side_effect_embedding(self, cui):
        
        """
        Get the embedding of a side effect based on its UMLS CUI.

        Parameters
        ----------
        cui : str
            The UMLS CUI of the side effect.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of size 768 representing the embedding of the side effect.
        """

        se_row = self.side_effects[self.side_effects['umls_cui_from_meddra'] == cui]
        if len(se_row) == 0:
            return np.zeros(768)
        return se_row.iloc[0][[str(i) for i in range(768)]].values.astype(float)

def process_quarter_data(processor, quarter, data_type):
    
    """
    Process a quarter of data and generate MLP input embeddings.

    Parameters
    ----------
    processor : EmbeddingProcessor
        The processor object to use for generating embeddings.
    quarter : str
        The quarter to process, e.g. "2015Q1".
    data_type : str
        The type of data to process, either "positive" or "negative".

    Returns
    -------
    None
    """
    input_path = f'dataset/evaluation_subset/subset_drug2-8_SE5-50/{quarter}_{data_type}_samples_condition123_SE_above_0.9.csv'
    output_path = f'dataset/MLP_input/{quarter}_{data_type}_embeddings.csv'
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    df = pd.read_csv(input_path)
    
    # Create the result DataFrame
    embeddings_data = []
    
    for _, row in tqdm(df.iterrows(), desc=f"Processing {quarter}_{data_type}"):
        drug_emb = processor.get_drugs_embedding(row['DrugBankID'])
        se_emb = processor.get_side_effect_embedding(row['SE_above_0.9'])
        
        # Concatenate the embeddings
        features = np.concatenate([drug_emb, se_emb])
        embeddings_data.append({
            'DrugBankID': row['DrugBankID'],
            'SE_above_0.9': row['SE_above_0.9'],
            'hyperedge_label': row['hyperedge_label'],
            **{f'feature_{i}': val for i, val in enumerate(features)}
        })
    
    # Save the embeddings
    os.makedirs('dataset/MLP_input', exist_ok=True)
    pd.DataFrame(embeddings_data).to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def main():
    """
    Process data for multiple quarters and generate MLP input embeddings.

    This function initializes an EmbeddingProcessor and iterates over a list
    of specified quarters and data types ('positive' and 'negative'). For each
    combination, it processes the data to generate embeddings and writes the
    results to CSV files.

    Returns
    -------
    None
    """

    processor = EmbeddingProcessor()
    
    # Define the quarter list
    quarters = ['2014Q3', '2014Q4', '2015Q1', '2015Q2', '2015Q3', '2015Q4',
                '2016Q1', '2016Q2', '2016Q3', '2016Q4', '2017Q1', '2017Q2',
                '2017Q3', '2017Q4', '2018Q1', '2018Q2', '2018Q3', '2018Q4',
                '2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1', '2020Q2',
                '2020Q3', '2020Q4', '2021Q1', '2021Q2', '2021Q3', '2021Q4',
                '2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2',
                '2023Q3', '2023Q4', '2024Q1', '2024Q2', '2024Q3']
    
    for quarter in quarters:
        for data_type in ['positive', 'negative']:
            process_quarter_data(processor, quarter, data_type)

if __name__ == "__main__":
    main()