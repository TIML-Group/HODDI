# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Set the environment variable
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Import libraries
import re
import pandas as pd
import matplotlib.pyplot as plt



# Import custom modules
from utils.visualization_utils import *

if __name__ == "__main__":
    
    ### GNN Visualization ###
    # Visualize training process 
    base_path = r"model\GNN\hetero_GraphConv\vanilla"
    visualize_training_process_for_clique_expansion(base_path)
    
    # Merge multiple training csv files
    input_dir = r'model\GNN\hetero_GraphConv\vanilla'
    merge_multiple_train_csvs(input_dir)
    
    ### MLP Visualization ###
    # Visualize training process 
    base_path = r"model\MLP\v1"
    visualize_training_process_for_MLP(base_path)
    
    # Merge multiple training csv files
    input_dir = r'model\MLP\v1'
    merge_multiple_train_csvs(input_dir)
    
    
    