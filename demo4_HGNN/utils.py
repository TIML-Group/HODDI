# Import libraries
import os
import re
import requests
from xml.etree import ElementTree as ET
import sys
import math
import random
import requests
import torch
import torch as t
import numpy as np
import pandas as pd
from tqdm import tqdm
import cobra
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
from rdkit import Chem
import deepchem as dc
from xml.etree import ElementTree as ET
from process_data import get_coefficient_and_reactant, create_neg_rxn
from torch_geometric.data import InMemoryDataset, Batch

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def suppress_cpp_warnings():
    """_summary_
    This is function is used to ignore the warnings from the native library like generate from c++
    Returns:
        _type_: the original stderr file descriptor
    """
    # save the current file descriptor
    stderr_fileno = sys.stderr.fileno()
    original_stderr = os.dup(stderr_fileno)

    # use /dev/null as the output so that no visible output
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fileno)

    return original_stderr, devnull

def restore_stderr(original_stderr, devnull):
    """
    Restor the original stderr file descriptor
    Args:
        original_stderr (_type_): _description_
        devnull (_type_): _description_
    """
    os.dup2(original_stderr, sys.stderr.fileno())
    # close the file descriptor
    os.close(devnull)
    os.close(original_stderr)

def create_neg_rxns(args):
    print('reading dataset----------------------------------------------')
    with open(f'./data/{args.train}/{args.train}_rxn_name_list.txt', 'r') as f:
        pos_rxn = [i.strip().replace('\n', '') for i in f.readlines()]
    pos_index, pos_metas, pos_nums, rxn_directions = get_coefficient_and_reactant(pos_rxn)
    pos_metas_smiles = pd.read_csv(f'./data/{args.train}/{args.train}_meta_count.csv')
    chebi_meta_filter = pd.read_csv('./data/pool/cleaned_chebi.csv')
    name_to_smiles = pd.concat(
        [chebi_meta_filter.loc[:, ['name', 'smiles']], pos_metas_smiles.loc[:, ['name', 'smiles']]])

    print('creating negative rxns --------------------------------------')
    neg_rxn = create_neg_rxn(pos_rxn, pos_metas_smiles, chebi_meta_filter, args.balanced, args.negative_ratio,
                             args.atom_ratio)
    neg_index, neg_metas, neg_nums, rxn_directions = get_coefficient_and_reactant(neg_rxn)
    all_metas = list(set(sum(pos_metas, []) + sum(neg_metas, [])))
    all_metas.sort()

    pos_matrix = np.zeros((len(all_metas), len(pos_rxn)))
    rxn_df = pd.DataFrame(pos_matrix, index=all_metas, columns=['p_' + str(i) for i in range(len(pos_rxn))])
    reaction_smiles = []
    for i in range(len(pos_index)):
        reactants = []
        products = []
        for j in range(len(pos_metas[i])):
            rxn_df.loc[pos_metas[i][j], 'p_' + str(i)] = float(pos_index[i][j])
            if j < pos_nums[i]:
                reactants.append(name_to_smiles[pos_metas[i][j] == name_to_smiles["name"]].smiles.values[0])
            else:
                products.append(name_to_smiles[pos_metas[i][j] == name_to_smiles["name"]].smiles.values[0])
        direction = rxn_directions[i]
        smiles = "+".join(reactants) + direction + "+".join(products)
        reaction_smiles.append(smiles)

    neg_matrix = np.zeros((len(all_metas), len(neg_rxn)))
    neg_df = pd.DataFrame(neg_matrix, index=all_metas, columns=['n_' + str(i) for i in range(len(neg_rxn))])
    for i in range(len(neg_index)):
        for j in range(len(neg_metas[i])):
            neg_df.loc[neg_metas[i][j], 'n_' + str(i)] = float(neg_index[i][j])
    label2rxn_df = pd.DataFrame(
        {'label': rxn_df.columns.to_list() + neg_df.columns.to_list(), 'rxn': pos_rxn + neg_rxn})

    return rxn_df, neg_df, name_to_smiles, label2rxn_df, reaction_smiles


def set_random_seed(seed=None):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int, optional): The random seed. Defaults to None.

    Returns:
        int: The random seed that was set.
    """
    
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    return seed


def get_filenames(path):
    """
    Return a list of filenames sorted in alphabetical order from a given path

    Parameters
    ----------
    path : str
        The path to the directory

    Returns
    -------
    list
        A list of filenames sorted in alphabetical order
    """
    
    return sorted(os.listdir(path))

def fold_and_reduce(data, target_dim=64):
    """
    Fold and reduce a list of molecular features to a target dimension.

    Parameters
    ----------
    data : list
        A list of molecular features
    target_dim : int, optional
        The target dimension to reduce to. Defaults to 64

    Returns
    -------
    np.array
        The reduced molecular features
    """
    
    results = []
    
    for row in data:
        result = reduce_item(row, target_dim)
        
        results.append(result)
    
    return np.array(results)

# -----molecular_graph_feature
def calculate_graph_feat(feat_mat, adj_list):
    """
    Calculate the graph features from a feature matrix and adjacency list.

    Parameters
    ----------
    feat_mat : np.ndarray
        Feature matrix where each row corresponds to a node's features.
    adj_list : list of lists
        Adjacency list representing the connections between nodes.

    Returns
    -------
    list
        A list containing:
        - feat_mat : np.ndarray
          The original feature matrix.
        - adj_index : np.ndarray
          A 2D array of shape (2, num_edges) representing the indices of connected nodes.
        - adj_mat : np.ndarray
          A symmetric adjacency matrix of shape (num_nodes, num_nodes).
    """

    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index, adj_mat]


def molecular_feature_extract(mol_data):
    """
    Extract molecular features from a list of molecular data.

    Parameters
    ----------
    mol_data : list
        A list of molecular data, where each element is a tuple containing a feature matrix and an adjacency list.

    Returns
    -------
    list
        A list of molecular features, where each element is a list containing the feature matrix, adjacency index, and adjacency matrix.
    """
    mol_data = pd.DataFrame(mol_data).T
    mol_feat = [[] for _ in range(len(mol_data))]
    for i in range(len(mol_feat)):
        feat_mat, adj_list = mol_data.iloc[i]
        mol_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return mol_feat

def reduce_item(row, target_dim=64):
    """
    Reduce a 1D array to a fixed length of 64 by padding and XOR operation.

    Parameters
    ----------
    row : np.ndarray
        A 1D array to be reduced.
    target_dim : int, optional
        The target dimension of the reduced array, by default 64.

    Returns
    -------
    np.ndarray
        A 1D array of shape (target_dim, ).
    """

    current_length = len(row)
        
    # calculate the length, make it the times of target_dim
    needed_length = ((current_length - 1) // target_dim + 1) * target_dim
    
    # padding with zeros
    if current_length < needed_length:
        row = np.pad(row, (0, needed_length - current_length), mode='constant')

    row = row.astype(np.int64)
    
    # reshape to (k, 64)
    reshaped_row = row.reshape(-1, target_dim)
    
    # xor each colume
    result = np.bitwise_xor.reduce(reshaped_row, axis=0)

    return result

def featurize_smiles(smiles):
    """
    Convert a list of SMILES strings into a dictionary of features and adjacency lists.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the featurized SMILES strings. Each row is a SMILES string, and contains two columns: 'atom_features' and 'adjacency_list'.
    """
    
    featurizer = dc.feat.ConvMolFeaturizer()
    features = pd.DataFrame()

    original_stderr, devnull = suppress_cpp_warnings()
    try:
        for i, smile_item in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile_item)
            # mol = Chem.AddHs(mol)
            mol_f = featurizer.featurize(mol)
            if mol_f:  
                atom_feat = mol_f[0].get_atom_features()
                adjacency = mol_f[0].get_adjacency_list()
                features[str(i)] = [atom_feat, adjacency]
    finally:
        restore_stderr(original_stderr, devnull)

    return features


def remove_rxn(model, name_list):
    """
    Remove metabolites from a model based on a list of names.

    Parameters
    ----------
    model : cobra.Model
        The model to remove metabolites from.
    name_list : list of str
        A list of metabolite names to remove from the model.

    Returns
    -------
    None
    """
    remove_list = []
    for i in range(len(model.metabolites)):
        meta = model.metabolites[i]
        if meta.name in name_list:
            continue
        remove_list.append(meta)
    model.remove_metabolites(remove_list, destructive=True)
    print(f'remove_rxn:{len(remove_list)}')


def get_data(path, sample):
    """
    Load a model from an SBML file located at path/sample.

    Parameters
    ----------
    path : str
        The directory containing the model file.
    sample : str
        The filename of the model file.

    Returns
    -------
    cobra.Model
        The loaded model.
    """
    
    model = cobra.io.read_sbml_model(path + '/' + sample)
    return model


def create_pool():
    """
    Create a metabolic model pool by merging individual GEMs (genome-scale metabolic models) with a reaction pool.

    This function reads metabolic models from XML files in a specified directory, merges them into a universal model pool,
    and writes the combined model to an output file. It also prints the total number of metabolites in the final model pool.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    print('-------------------------------------------------------')
    print('merging GEMs with reaction pool...')
    path = 'data/gems/xml-file'
    namelist = get_filenames(path)
    model_pool = cobra.io.read_sbml_model('./data/pool/universe.xml')
    pool_df = create_stoichiometric_matrix(model_pool, array_type='DataFrame')
    for sample in namelist:
        if sample.endswith('xml'):
            model = get_data(path, sample)
            model_pool.merge(model)
    cobra.io.write_sbml_model(model_pool, './results/bigg/comb_universe-fliter.xml')
    print(f'create pool done! The total number of metabolites is {len(model_pool.metabolites)}')


def get_data_from_pool(path, sample, model_pool_df):
    """
    Load a model from a pool of reactions and get the corresponding subset of a stoichiometric matrix.

    Parameters
    ----------
    path : str
        The directory containing the model file.
    sample : str
        The filename of the model file.
    model_pool_df : pd.DataFrame
        The stoichiometric matrix of the universal model pool.

    Returns
    -------
    tuple
        A tuple containing two DataFrames. The first element is the subset of the stoichiometric matrix corresponding to the model.
        The second element is the columns of the stoichiometric matrix that are not in the model.
    """
    if os.path.exists(path + '/reactions_w_gene_reaction_rule.csv'):
        rxns_df = pd.read_csv(path + '/reactions_w_gene_reaction_rule.csv')
        rxns = rxns_df.reaction[rxns_df.id == sample[:-4]].to_numpy()
    else:
        model = get_data(path, sample)
        rxns = np.array([rxn.id for rxn in model.reactions])
    model_df = model_pool_df[rxns]
    cols2use = model_pool_df.columns.difference(model_df.columns)
    return model_df, model_pool_df[cols2use]


def create_neg_incidence_matrix(incidence_matrix):
    """
    Create a negative incidence matrix by randomly sampling half of the nodes in each edge and swapping them with nodes from the complement set.

    Parameters
    ----------
    incidence_matrix : torch.Tensor
        The incidence matrix of the hypergraph.

    Returns
    -------
    torch.Tensor
        The negative incidence matrix.
    """
    incidence_matrix_neg = t.zeros(incidence_matrix.shape)
    for i, edge in enumerate(incidence_matrix.T):
        nodes = t.where(edge)[0]
        nodes_comp = t.tensor(list(set(range(len(incidence_matrix))) - set(nodes.tolist())))
        edge_neg_l = t.tensor(np.random.choice(nodes, math.floor(len(nodes) * 0.5), replace=False))
        edge_neg_r = t.tensor(
            np.random.choice(nodes_comp, len(nodes) - math.floor(len(nodes) * 0.5), replace=False))
        edge_neg = t.cat((edge_neg_l, edge_neg_r))
        incidence_matrix_neg[edge_neg, i] = 1
    return incidence_matrix_neg


def hyperlink_score_loss(y_pred, y):
    """
    Hyperlink score loss function.

    Parameters
    ----------
    y_pred : torch.Tensor
        The predicted scores of the hyperlinks.
    y : torch.Tensor
        The true labels of the hyperlinks.

    Returns
    -------
    torch.Tensor
        The loss of the hyperlink score prediction.
    """
    negative_score = t.mean(y_pred[y == 0])
    logistic_loss = t.log(1 + t.exp(negative_score - y_pred[y == 1]))
    loss = t.mean(logistic_loss)
    return loss


def create_label(incidence_matrix_pos, incidence_matrix_neg):
    """
    Create a label tensor for hyperlink prediction from positive and negative incidence matrices.

    Parameters
    ----------
    incidence_matrix_pos : torch.Tensor
        The positive incidence matrix.
    incidence_matrix_neg : torch.Tensor
        The negative incidence matrix.

    Returns
    -------
    torch.Tensor
        The label tensor for hyperlink prediction.
    """
    
    y_pos = t.ones(len(incidence_matrix_pos.T))
    y_neg = t.zeros(len(incidence_matrix_neg.T))
    return t.cat((y_pos, y_neg))


def getGipKernel(y, trans, gamma):
    """
    Compute the GIP kernel matrix from a given matrix of node features.

    Parameters
    ----------
    y : torch.Tensor
        The matrix of node features.
    trans : bool
        Whether to transpose the matrix.
    gamma : float
        The hyperparameter for the GIP kernel.

    Returns
    -------
    torch.Tensor
        The GIP kernel matrix.
    """
    if trans:
        y = y.T
    krnl = t.mm(y, y.T)
    krnl = krnl / t.mean(t.diag(krnl))
    krnl = t.exp(-kernelToDistance(krnl) * gamma)
    return krnl


def kernelToDistance(k):
    """
    Convert a kernel matrix to a distance matrix.

    Parameters
    ----------
    k : torch.Tensor
        A square kernel matrix.

    Returns
    -------
    torch.Tensor
        A distance matrix derived from the kernel matrix, where each element
        represents the distance between corresponding nodes.
    """

    di = t.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d


def cosine_kernel(tensor_1, tensor_2):
    """
    Compute the cosine similarity between each row of tensor_1 and tensor_2.

    Parameters
    ----------
    tensor_1 : torch.Tensor
        A tensor of shape (n_samples, n_features) where each row is a sample.
    tensor_2 : torch.Tensor
        A tensor of shape (n_samples, n_features) where each row is a sample.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_samples,) containing the cosine similarity between
        each row of tensor_1 and tensor_2.
    """
    return t.DoubleTensor([t.cosine_similarity(tensor_1[i], tensor_2, dim=-1).tolist() for i in
                           range(tensor_1.shape[0])])

def min_max_normalize_cuda(tensor):

    """
    Normalize a CUDA tensor to the range [0, 1] by subtracting the minimum and
    dividing by the range.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to normalize.

    Returns
    -------
    normalized_tensor : torch.Tensor
        The normalized tensor.
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def min_max_offset_cuda(tensor):

    """
    Shift a CUDA tensor to have a minimum value of 0 by adding the tensor's
    minimum value to all elements.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to shift.

    Returns
    -------
    shifted_tensor : torch.Tensor
        The shifted tensor.
    """
    
    min_val = torch.min(tensor)


    return tensor + min_val

def monitor_gradients(model):

    """
    Monitor the gradients of model parameters during training and detect vanishing gradients.

    Parameters
    ----------
    model : torch.nn.Module
        The model containing parameters to monitor.

    This function iterates over the model's parameters, checking if they require gradient computation
    and if their gradients are available. It computes the norm of each parameter's gradient and prints
    a warning message if the gradient norm is below a specified threshold (1e-6), indicating potential
    vanishing gradients.
    """

    for name, param in model.named_parameters():  
        if param.requires_grad and param.grad is not None:  
            grad_norm = param.grad.norm()  
            if grad_norm < 1e-6:
                print(f"Warning: Gradient vanishing detected in {name} layer")


# Regular expression pattern to match chemical names
pattern = re.compile(r"(?<!\d)\s*s_\d+\[\w+\]")

def extract_chemicals(reaction):
    

    """
    Extract chemical names from a reaction string.

    Parameters
    ----------
    reaction : str
        The reaction string.

    Returns
    -------
    reactant_names : list
        The list of reactant chemical names.
    product_names : list
        The list of product chemical names.
    direction : str
        The reaction direction ('=>' or '<=>').
    """
    
    # Get reaction direction
    direction = '=>'
    if '<=>' in reaction:
        direction = '<=>'

    # Split reaction into reactants and products
    reactants, products = reaction.split(direction)
    
    # Extract chemical names from reactants and products
    reactant_names = pattern.findall(reactants)
    product_names = pattern.findall(products)
    
    # Remove brackets and their contents from each chemical name
    reactant_names = [re.sub(r"\[\w+\]", "", name.strip()) for name in reactant_names]
    product_names = [re.sub(r"\[\w+\]", "", name.strip()) for name in product_names]
    
    return reactant_names, product_names, direction



def convert_mol_to_smiles(mol_data):
    """
    Convert a mol format string to a SMILES string using RDKit.

    Parameters
    ----------
    mol_data : str
        The mol format string.

    Returns
    -------
    str or None
        The SMILES string if successful, otherwise None.
    """
    try:
        mol = Chem.MolFromMolBlock(mol_data) 
        if mol:
            smiles = Chem.MolToSmiles(mol)  
            return smiles
        else:
            print("RDKit failed to parse the mol data.")
    except Exception as e:
        print(f"Error converting mol to SMILES: {e}")
    return None

def remove_namespace(tree):
    """
    Remove namespace from an ElementTree object.

    Parameters
    ----------
    tree : ElementTree
        The ElementTree object to remove namespace from.

    Returns
    -------
    ElementTree
        The ElementTree object with namespace removed.
    """

    for elem in tree.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return tree

def get_smiles_from_chebi(chebi_id):
    """
    Fetch SMILES string for a given ChEBI ID from the ChEBI Web Service.

    Parameters
    ----------
    chebi_id : str
        The ChEBI ID to fetch the SMILES string for.

    Returns
    -------
    str or None
        The SMILES string if successful, otherwise None.
    """
    url = f'https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity?chebiId={chebi_id}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse XML and remove namespace
            tree = ET.ElementTree(ET.fromstring(response.content))
            tree = remove_namespace(tree)
            root = tree.getroot()

            smiles = None

            # Query the root element for the 'smiles' element
            smiles_element = root.find('.//smiles')
            if smiles_element is not None:
                smiles = smiles_element.text

            # Query the root element for the 'mol' element if 'smiles' element is not found
            if not smiles:
                mol_data = None
                for structure in root.findall('.//ChemicalStructures'):
                    structure_type = structure.find('type').text
                    default_structure = structure.find("defaultStructure").text
                    if structure_type == 'mol' and default_structure == 'true':  
                        mol_data = structure.find('structure').text
                        break
                
                if mol_data:
                    # Convert mol format to SMILES using RDKit
                    smiles = convert_mol_to_smiles(mol_data)  
                    if smiles:
                        print(f"Converted mol to SMILES for ChEBI ID {chebi_id}: {smiles}")
                    else:
                        print(f"Failed to convert mol to SMILES for ChEBI ID {chebi_id}.")
                else:
                    print(f"No SMILES or mol structure found for ChEBI ID {chebi_id}.")
            
            return smiles
        else:
            print(f"Failed to fetch data for ChEBI ID {chebi_id}. Status code: {response.status_code}")
    except ET.ParseError as parse_err:
        print(f"Error parsing XML for ChEBI ID {chebi_id}: {parse_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error while fetching data for ChEBI ID {chebi_id}: {req_err}")
    
    return None

def get_smiles_from_kegg(kegg_id):
    """
    Fetch the SMILES string for a given KEGG ID using the KEGG REST API.

    Parameters
    ----------
    kegg_id : str
        The KEGG ID to fetch the SMILES string for.

    Returns
    -------
    str or None
        The SMILES string if successful, otherwise None.
    """
    url = f'http://rest.kegg.jp/get/{kegg_id}/mol'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            molfile = response.text
            from rdkit import Chem
            mol = Chem.MolFromMolBlock(molfile)
            if mol:
                smiles = Chem.MolToSmiles(mol)
                return smiles
    except Exception as e:
        print(f"Error fetching SMILES from KEGG for ID {kegg_id}: {e}")
    return None