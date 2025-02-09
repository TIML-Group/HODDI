# Import libraries
import os
import utils
import torch
import gensim
import numpy as np
from rdkit import Chem
from mol2vec.features import mol2alt_sentence


# Set model path
current_dir = os.path.basename(os.getcwd())

if current_dir == "fba":
    model_path = "../model_300dim.pkl"
else:
    model_path = "./model_300dim.pkl"

# Load Word2Vec model
try:
    mol2vec_model = gensim.models.Word2Vec.load(model_path)
    print(f"Model successfully loaded from {model_path}")
except FileNotFoundError:
    print(f"Model not found at {model_path}")
except Exception as e:
    print(f"Error: {e}")

def get_average_vector(sentence, model, unseen=None):
    """
    Calculate the average vector of all word vectors in a sentence.
    :param sentence: List of words, derived from the molecule's "sentence".
    :param model: Word vector model, such as Word2Vec or FastText.
    :param unseen: Handling for unseen words (if a word does not exist in the model).
    :return: The average vector of the words in the sentence.
    """
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])  # Append the word's vector if it exists in the model
        elif unseen:
            vectors.append(model.wv[unseen])  # Append the unseen word vector if provided
    if not vectors:
        return np.zeros(model.vector_size)  # Return a zero vector if no vectors are found
    return np.mean(vectors, axis=0)  # Return the average of the vectors

def smiles_to_mol2vec(smiles, radius=1):
    """
    Convert SMILES strings to mol2vec vectors.
    :param smiles: List of SMILES strings.
    :param model: Pre-trained mol2vec model.
    :param radius: Radius used (default is 1, consistent with the model's training).
    :return: 300-dimensional mol2vec vector.
    """
    mol_feat = []

    # Suppress C++ warnings
    original_stderr, devnull = utils.suppress_cpp_warnings()
    try:
        for i, smiles_item in enumerate(smiles):
            mol = Chem.MolFromSmiles(smiles_item)  # Convert SMILES to a molecule
            # mol = Chem.AddHs(mol)  # Optionally add hydrogens to the molecule
            # Convert the molecule to a "sentence"
            sentence = mol2alt_sentence(mol, radius)
            # Calculate the average vector of the sentence
            mol2vec_vector = get_average_vector(sentence, mol2vec_model, unseen='UNK').reshape(1, -1)
            mol_feat.append(mol2vec_vector)  # Append the resulting vector
    finally:
        utils.restore_stderr(original_stderr, devnull)  # Restore the original stderr

    mol_feat = torch.tensor(np.vstack(mol_feat), dtype=torch.float32)  # Stack the vectors into a tensor
    return mol_feat  # Return the resulting mol2vec vectors