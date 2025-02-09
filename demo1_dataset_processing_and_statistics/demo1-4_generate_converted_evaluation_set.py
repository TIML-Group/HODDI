# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Import libraries
import shutil
import numpy as np
import pandas as pd
from itertools import combinations
import xml.etree.ElementTree as ET
from collections import defaultdict

# Import custom modules
from utils.data_utils import *

# Main function
if __name__ == "__main__":
    
    # Generate a dictionary mapping BankDrugID to SMILES
    generate_BankDrugID2SMILES()
    
    # Convert HGNN dataset to GNN dataset by clique expansion
    input_dir = 'dataset/evaluation_subset/subset_drug2-8_SE5-50'
    output_dir = 'dataset/converted_evaluation_subset/subset_drug2-8_SE5-50/clique_expansion'
    hgnn2gnn_clique_expansion_in_a_given_directory(input_dir, output_dir)
    
    