# Import libraries
import utils
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.utils.data as Data
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HyperGraphDataset(InMemoryDataset):
    """ This class is used for storing molecular feature consistent of atoms features and their relatiionship.
    Args:
        root (string): root
        dataset (string): the name of this dataset
        transform (callable): inherited from InMemoryDataset
        pre_transform (callable): inherited from InMemortyDataset
        graphs_dict (Any): input data
    """
    def __init__(self, root='.', dataset='data', all_smiles=None, transform=None, pre_transform=None, dttype=None):
        super(HyperGraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(all_smiles)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, all_smiles):
        data_list = []
        # Process each SMILES string (each molecule represents a hyperedge)
        for smiles in all_smiles:
            node_features, edge_features, incidence_mat = mol_to_hypergraph(smiles)
            HGData = DATA.Data(x=node_features, edge_attr=edge_features, edge_index=incidence_mat)
            data_list.append(HGData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    items = []
    for i, item in enumerate(data_list):
        items.append(item)
    # batchA = Batch.from_data_list([data for data in data_list])
    batchA = Batch.from_data_list(items)
    return batchA.to(device)

def mol_to_hypergraph(smiles, radius = 2, node_features_dim=64, edge_features_dim=64):
    mol = Chem.MolFromSmiles(smiles)
    
    original_stderr, devnull = utils.suppress_cpp_warnings()
    try:
    # Generate node features (using Morgan fingerprints for simplicity)
        node_features = []
        for atom in mol.GetAtoms():
            node_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=node_features_dim, fromAtoms=[atom.GetIdx()])
            node_features.append(np.array(node_fp))
        node_features = torch.FloatTensor(node_features).to(device)
        
        # Generate edge features and incidence matrix
        edge_features = []
        incidence_matrix = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=edge_features_dim, fromAtoms=[start, end])
            edge_features.append(np.array(bond_fp))
            incidence_matrix.append([start, end])
        # Handle cases where there are no bonds
        if len(edge_features) == 0:
            # If no bonds, create a dummy edge feature of all zeros
            edge_features = torch.zeros(1, edge_features_dim).to(device)
        else:
            edge_features = torch.FloatTensor(edge_features).to(device)
        
        # Handle cases where there is no incidence matrix (no bonds)
        if len(incidence_matrix) == 0:
            # If no bonds, create a dummy incidence matrix of all zeros
            incidence_matrix = torch.zeros(1, 2, dtype=torch.long).to(device)
        else:
            incidence_matrix = torch.LongTensor(incidence_matrix).to(device)
    finally:
        utils.restore_stderr(original_stderr, devnull)
    
    return node_features, edge_features, incidence_matrix

def smiles_to_hyper_graph2(all_smiles, radius = 2, node_features_dim=64, edge_features_dim=64):
    """
    Convert a list of SMILES strings into a PyTorch Geometric Data.DataLoader of HyperGraphDataset.
    
    Parameters
    ----------
    all_smiles : list of str
        A list of SMILES strings representing chemical compounds.
    radius : int, optional
        The radius of the Morgan fingerprint, by default 2.
    node_features_dim : int, optional
        The dimension of the node features, by default 64.
    edge_features_dim : int, optional
        The dimension of the edge features, by default 64.
    
    Returns
    -------
    torch.utils.data.DataLoader
        A PyTorch Geometric Data.DataLoader containing the HyperGraphDataset.
    """
    data_set = Data.DataLoader(dataset=HyperGraphDataset(all_smiles=all_smiles),
                                   collate_fn=collate, batch_size=len(all_smiles), shuffle=False)
    return data_set

def smiles_to_hyper_graph(all_smiles, radius = 2, node_features_dim=64, edge_features_dim=64):
    """
    Convert a list of SMILES strings into a list of PyTorch Geometric Data.Data.
    
    Parameters
    ----------
    all_smiles : list of str
        A list of SMILES strings representing chemical compounds.
    radius : int, optional
        The radius of the Morgan fingerprint, by default 2.
    node_features_dim : int, optional
        The dimension of the node features, by default 64.
    edge_features_dim : int, optional
        The dimension of the edge features, by default 64.
    
    Returns
    -------
    list of torch_geometric.data.data.Data
        A list of Data objects, each containing the node features, edge features, and incidence matrix for a hypergraph.
    """
    data_list = []
    # Process each SMILES string (each molecule represents a hyperedge)
    for smiles in tqdm(all_smiles):
        node_features, edge_features, incidence_mat = mol_to_hypergraph(smiles)
        HGData = DATA.Data(x=node_features, edge_attr=edge_features, edge_index=incidence_mat)
        data_list.append(HGData)
    return data_list