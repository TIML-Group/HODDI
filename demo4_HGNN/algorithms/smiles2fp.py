# Import libraries
import utils
import torch
import numpy as np
import deepchem as dc
import rdkit.Chem as Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles, radius=2, nBits=64):
    """
    Convert a list of SMILES strings to their Morgan fingerprints.

    Parameters
    ----------
    smiles: list
        List of SMILES strings.
    radius: int, optional
        The radius of the Morgan fingerprint. Defaults to 2.
    nBits: int, optional
        The number of bits in the fingerprint. Defaults to 64.

    Returns
    -------
    tensor: torch.Tensor
        A tensor of shape `(n_samples, nBits)` containing the Morgan fingerprints.
    """
    features = []

    original_stderr, devnull = utils.suppress_cpp_warnings()
    try:
        for smile_item in smiles:
            mol = Chem.MolFromSmiles(smile_item)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=False)
            fp_bits = fp.ToBitString()
            finger_print = np.array(list(map(int, fp_bits))).astype(float).reshape(1, -1)
            features.append(finger_print)
    finally:
        utils.restore_stderr(original_stderr, devnull)
    return torch.tensor(np.vstack(features), dtype=torch.float32)

def Tanimoto_smi(smiles_list):
    """
    Calculate the Tanimoto similarity matrix for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list
        A list of SMILES strings for which the Tanimoto similarity is to be calculated.

    Returns
    -------
    smi : np.ndarray
        A symmetric matrix of size `(n, n)` where `n` is the number of SMILES strings.
        Each element `smi[i, j]` represents the Tanimoto similarity between the
        fingerprints of the `i`-th and `j`-th SMILES.
    """

    def _compute(data_1, data_2):
        norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
        norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
        prod = data_1.dot(data_2.T)
        return prod / (norm_1 + norm_2.T - prod)

    fps = smiles_to_fingerprint(smiles_list)
    smi = np.ones((len(fps), len(fps)))
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            smi[i, j] = _compute(fps[i], fps[j])
            smi[j, i] = smi[i, j]
    return smi