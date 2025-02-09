# Import libraries
import utils
import torch
import numpy as np
from tqdm import tqdm
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints

# Initialize the RXNBERTFingerprintGenerator
model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

def smiles_to_reaction_fingerprint(smiles):
    """
    Convert a list of SMILES strings representing chemical reactions into
    reaction fingerprint vectors.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings representing chemical reactions.

    Returns
    -------
    torch.Tensor
        A tensor containing the feature vectors for each SMILES string. The
        feature vector dimension is 1024.
    """
    features = []

    original_stderr, devnull = utils.suppress_cpp_warnings()
    try:
        # for smile_item in tqdm(smiles):
        for smile_item in smiles:
            fp = rxnfp_generator.convert(smile_item)
            features.append(fp)
    finally:
        utils.restore_stderr(original_stderr, devnull)
    return torch.tensor(np.vstack(features), dtype=torch.float32)

# example_rxn = "Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1"

# fp = rxnfp_generator.convert(example_rxn)
# print(len(fp))
# print(fp[:5])
