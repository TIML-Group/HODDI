# Import libraries
import utils
import torch
import torch.utils.data as Data
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphDataset(InMemoryDataset):
    """ This class is used for storing molecular feature consistent of atoms features and their relatiionship.
    Args:
        root (string): root
        dataset (string): the name of this dataset
        transform (callable): inherited from InMemoryDataset
        pre_transform (callable): inherited from InMemortyDataset
        graphs_dict (Any): input data
    """
    def __init__(self, root='.', dataset='data', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

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

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features = torch.Tensor(data_mol[0])
            edge_index = torch.LongTensor(data_mol[1])
            GCNData = DATA.Data(x=features, edge_index=edge_index)
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(device)


def smiles_to_GCN(smiles):
    """
    Convert a list of SMILES strings into a PyTorch Geometric Data loader.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader containing a single batch of molecular features
    """
    smiles_feat = utils.featurize_smiles(smiles)
    smiles_feat = utils.molecular_feature_extract(smiles_feat)
    data_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=smiles_feat),
                                   collate_fn=collate, batch_size=len(smiles_feat), shuffle=False)

    return data_set