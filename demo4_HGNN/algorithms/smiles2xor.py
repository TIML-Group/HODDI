# Import libraries
import utils
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def smiles_to_atom_xor(smiles, emb_dim):
    """
    Convert a list of SMILES strings to atom features (using XOR operation).
    
    Parameters
    ----------
    smiles : list
        A list of SMILES strings.
    emb_dim : int
        The dimension of the embedding space.
    
    Returns
    -------
    features : torch.tensor
        A tensor of shape (n, emb_dim), where n is the number of atoms in the SMILES strings.
    """
    mol_feat = utils.featurize_smiles(smiles)
    mol_feat = pd.DataFrame(mol_feat).T
    features = [[] for _ in range(len(mol_feat))]
    for i in range(len(features)):
        feat_mat, _ = mol_feat.iloc[i]
        atom_feat = feat_mat[0].flatten()

        atom_feat = np.array(atom_feat).flatten()
        atom_feat = atom_feat.astype(np.int64)
        atom_feat = utils.reduce_item(atom_feat, emb_dim)
        features[i] = atom_feat.flatten()

    features = np.vstack(features)
    return torch.tensor(features, dtype=torch.float32)