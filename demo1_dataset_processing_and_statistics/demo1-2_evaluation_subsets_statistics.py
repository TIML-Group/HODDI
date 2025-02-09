# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
import ast
import logging


# Import custom modules
from utils.statistics_utils import *

# Main function
if __name__ == "__main__":
    ### Part 2. Quarterly/yearly/overall statistics for positive and negative samples in HGNN dataset
    
    # 1. Overall statistics: 2014Q3_2024Q3 (41 Quarters)
    # 1-1. Statistics: a) Number of Drugs per Record; b) SE distribution
    df_paths_overall = [r'dataset\condition123subsets\positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv',r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv']
    output_dir = r'dataset\condition123subsets\statistics\2014Q3_2024Q3'
    
    for df_path in df_paths_overall:
        # a) Number of Drugs per Record
        plot_drug_statistics(df_path, output_dir) 
        # b) SE distribution
        plot_se_statistics(df_path, output_dir) 
        
    
    # 1-2. Stratified statistics: a) Number of Drugs per Record; b) SE distribution
    
    # a) Number of Drugs per Record
    # a-1. Drugs in positive samples
    input_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\positive_samples_drug_statistics.csv'
    output_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\stratified_statistics\drug_positive_samples.csv'
    drug_positive_and_negative_samples_stratified_statistics(input_csv_path, output_csv_path)
    
    # a-2. Drugs in negative samples
    input_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\negative_samples_drug_statistics.csv'
    output_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\stratified_statistics\drug_negative_samples.csv'
    drug_positive_and_negative_samples_stratified_statistics(input_csv_path, output_csv_path)
    
    # b) SE distribution
    # b-1. SE in positive samples
    input_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\positive_samples_SE_statistics.csv'
    output_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\stratified_statistics\SE_positive_samples.csv'
    se_positive_samples_stratified_statistics(input_csv_path, output_csv_path)
    
    # b-2. SE in negative samples
    input_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\negative_samples_SE_statistics.csv'
    output_csv_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\stratified_statistics\SE_negative_samples.csv'
    se_negative_samples_stratified_statistics(input_csv_path, output_csv_path)
    
    # Zoom-in picture for long-tailed distribution (b-1. SE in positive samples)
    zoom_in_se_positive_samples()
    
    # 2. Quarterly statistics: 2014Q3_2024Q3 (41 Quarters)
    os.rename(r'dataset\condition123subsets\statistics\num_stratified_se.csv', r'dataset\condition123subsets\statistics\se_stratified_per_quarter.csv')
    modify_and_save_csv1(r'dataset\condition123subsets\statistics\se_stratified_per_quarter.csv')
    
    os.rename(r'dataset\condition123subsets\statistics\summary.csv',r'dataset\condition123subsets\statistics\condition123_stratified_per_quarter.csv')
    modify_and_save_csv2(r'dataset\condition123subsets\statistics\condition123_stratified_per_quarter.csv')
    
    
