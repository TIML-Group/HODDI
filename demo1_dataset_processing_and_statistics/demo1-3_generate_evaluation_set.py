# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Import libraries
import pandas as pd
import shutil

# Import custom modules
from utils.data_utils import *

# Main function
if __name__ == "__main__":

    generate_subset_for_merged_positive_samples(min_drug_length=2, max_drug_length=8, min_se_count=5, max_se_count=50) #  21503 rows
    generate_subset_for_merged_positive_samples(min_drug_length=2, max_drug_length=16, min_se_count=5, max_se_count=50) # 24777 rows
    generate_subset_for_merged_positive_samples(min_drug_length=2, max_drug_length=16, min_se_count=5, max_se_count=100) # 38430 rows

    # Add row index in merged negative samples
    df = pd.read_csv(r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv')
    df['row_index'] = range(1, len(df) + 1)
    df.to_csv(r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv', index=False)
    
    
    fetch_negative_samples(positive_csv_path= r'dataset\evaluation_subset\subset_drug2-8_SE5-50\positive_samples_2014Q3_2024Q3_step6.csv')
    fetch_negative_samples(positive_csv_path= r'dataset\evaluation_subset\subset_drug2-16_SE5-50\positive_samples_2014Q3_2024Q3_step6.csv')
    fetch_negative_samples(positive_csv_path= r'dataset\evaluation_subset\subset_drug2-16_SE5-100\positive_samples_2014Q3_2024Q3_step6.csv')
    
    process_dataset_folder(r"dataset\evaluation_subset\subset_drug2-8_SE5-50")
    process_dataset_folder(r"dataset\evaluation_subset\subset_drug2-16_SE5-50")
    process_dataset_folder(r"dataset\evaluation_subset\subset_drug2-16_SE5-100")


