# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import libraries
import os
import gc
import shutil
import pickle
import zipfile
from tqdm import tqdm
import xml.etree.ElementTree as ET
from datetime import date, datetime
from zipfile_deflate64 import ZIP_DEFLATED64, ZipFile
import pandas as pd
import numpy as np
from datetime import datetime
import ast
import glob
import re
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

# Import custom modules
from utils.data_utils import *
from utils.statistics_utils import *

# Main function
if __name__ == "__main__":
    
    ### Part 1. Convert downloaded XML files to filtered HGNN dataset
    
    # Rename and extract zip files
    base_path = r"dataset\xml_download" # Set your own path to the original dataset folder
    try:
        rename_and_extract_zip(base_path)
        print("Processing completed!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
        
    ## Convert XML to PK files (2012Q4-2024Q3)
    
    # Set xml_batches of XML files
    
    xml_batches = [
        # 2012 Q4
        ['dataset/xml_download/faers_xml_2012Q4/xml/ADR12Q4.xml'],
        # 2013 Q1
        ['dataset/xml_download/faers_xml_2013Q1/xml/ADR13Q1.xml'],
        # 2013 Q2
        ['dataset/xml_download/faers_xml_2013Q2/xml/ADR13Q2.xml'],
        # 2013 Q3
        ['dataset/xml_download/faers_xml_2013Q3/xml/ADR13Q3.xml'],
        # 2013 Q4
        ['dataset/xml_download/faers_xml_2013Q4/xml/ADR13Q4.xml'],
        # 2014 Q1
        ['dataset/xml_download/faers_xml_2014Q1/xml/ADR14Q1.xml'],
        # 2014 Q2
        ['dataset/xml_download/faers_xml_2014Q2/xml/ADR14Q2.xml'],
        # 2014 Q3
        ['dataset/xml_download/faers_xml_2014Q3/xml/ADR14Q3.xml'],
        # 2014 Q4
        ['dataset/xml_download/faers_xml_2014Q4/xml/ADR14Q4.xml'],
        # 2015 Q1
        ['dataset/xml_download/faers_xml_2015Q1/xml/ADR15Q1.xml'],
        # 2015 Q2
        ['dataset/xml_download/faers_xml_2015Q2/xml/ADR15Q2.xml'],
        # 2015 Q3
        ['dataset/xml_download/faers_xml_2015Q3/xml/ADR15Q3.xml'],
        # 2015 Q4
        ['dataset/xml_download/faers_xml_2015Q4/xml/ADR15Q4.xml'],
        # 2016 Q1        
        ['dataset/xml_download/faers_xml_2016Q1/xml/ADR16Q1.xml'],
        # 2016 Q2
        ['dataset/xml_download/faers_xml_2016Q2/xml/ADR16Q2.xml'],
        # 2016 Q3
        ['dataset/xml_download/faers_xml_2016Q3/xml/ADR16Q3.xml'],
        # 2016 Q4
        ['dataset/xml_download/faers_xml_2016Q4/xml/ADR16Q4.xml'],
        # 2017 Q1
        ['dataset/xml_download/faers_xml_2017Q1/xml/ADR17Q1.xml'],
        # 2017 Q2
        ['dataset/xml_download/faers_xml_2017Q2/xml/ADR17Q2.xml'],
        # 2017 Q3
        ['dataset/xml_download/faers_xml_2017Q3/xml/ADR17Q3.xml'],
        # 2017 Q4
        ['dataset/xml_download/faers_xml_2017Q4/xml/ADR17Q4.xml'],
        # 2018 Q1
        ['dataset/xml_download/faers_xml_2018Q1/xml/ADR18Q1.xml'],  
        # 2018 Q2
        ['dataset/xml_download/faers_xml_2018Q2/xml/1_ADR18Q2_format.xml',
        'dataset/xml_download/faers_xml_2018Q2/xml/2_ADR18Q2_format.xml',
        'dataset/xml_download/faers_xml_2018Q2/xml/3_ADR18Q2_format.xml'],
        # 2018 Q3
        ['dataset/xml_download/faers_xml_2018Q3/xml/1_ADR18Q3.xml',
        'dataset/xml_download/faers_xml_2018Q3/xml/2_ADR18Q3.xml',
        'dataset/xml_download/faers_xml_2018Q3/xml/3_ADR18Q3.xml'],
        # 2018 Q4
        ['dataset/xml_download/faers_xml_2018Q4/xml/1_ADR18Q4.xml',
        'dataset/xml_download/faers_xml_2018Q4/xml/2_ADR18Q4.xml',
        'dataset/xml_download/faers_xml_2018Q4/xml/3_ADR18Q4.xml'],
        # 2019 Q1
        ['dataset/xml_download/faers_xml_2019Q1/xml/1_ADR19Q1.xml',
        'dataset/xml_download/faers_xml_2019Q1/xml/2_ADR19Q1.xml',
        'dataset/xml_download/faers_xml_2019Q1/xml/3_ADR19Q1.xml'],
        # 2019 Q2
        ['dataset/xml_download/faers_xml_2019Q2/xml/1_ADR19Q2.xml',
        'dataset/xml_download/faers_xml_2019Q2/xml/2_ADR19Q2.xml',
        'dataset/xml_download/faers_xml_2019Q2/xml/3_ADR19Q2.xml'],
        # 2019 Q3
        ['dataset/xml_download/faers_xml_2019Q3/xml/1_ADR19Q3.xml',
        'dataset/xml_download/faers_xml_2019Q3/xml/2_ADR19Q3.xml',
        'dataset/xml_download/faers_xml_2019Q3/xml/3_ADR19Q3.xml'],
        # 2019 Q4
        ['dataset/xml_download/faers_xml_2019Q4/xml/1_ADR19Q4.xml',
        'dataset/xml_download/faers_xml_2019Q4/xml/2_ADR19Q4.xml',
        'dataset/xml_download/faers_xml_2019Q4/xml/3_ADR19Q4.xml'],
        # 2020 Q1
        ['dataset/xml_download/faers_xml_2020Q1/xml/1_ADR20Q1.xml',
        'dataset/xml_download/faers_xml_2020Q1/xml/2_ADR20Q1.xml',
        'dataset/xml_download/faers_xml_2020Q1/xml/3_ADR20Q1.xml'],
        # 2020 Q2
        ['dataset/xml_download/faers_xml_2020Q2/xml/1_ADR20Q2.xml',
        'dataset/xml_download/faers_xml_2020Q2/xml/2_ADR20Q2.xml',
        'dataset/xml_download/faers_xml_2020Q2/xml/3_ADR20Q2.xml'],
        # 2020 Q3
        ['dataset/xml_download/faers_xml_2020Q3/xml/1_ADR20Q3.xml',
        'dataset/xml_download/faers_xml_2020Q3/xml/2_ADR20Q3.xml',
        'dataset/xml_download/faers_xml_2020Q3/xml/3_ADR20Q3.xml'],
        # 2020 Q4
        ['dataset/xml_download/faers_xml_2020Q4/xml/1_ADR20Q4.xml',
        'dataset/xml_download/faers_xml_2020Q4/xml/2_ADR20Q4.xml',
        'dataset/xml_download/faers_xml_2020Q4/xml/3_ADR20Q4.xml'],
        # 2021 Q1
        ['dataset/xml_download/faers_xml_2021Q1/XML/1_ADR21Q1_format.xml',
        'dataset/xml_download/faers_xml_2021Q1/XML/2_ADR21Q1_format.xml',
        'dataset/xml_download/faers_xml_2021Q1/XML/3_ADR21Q1_format.xml'],
        # 2021 Q2
        ['dataset/xml_download/faers_xml_2021Q2/XML/1_ADR21Q2.xml',
        'dataset/xml_download/faers_xml_2021Q2/XML/2_ADR21Q2.xml',
        'dataset/xml_download/faers_xml_2021Q2/XML/3_ADR21Q2.xml'],
        # 2021 Q3
        ['dataset/xml_download/faers_xml_2021Q3/XML/1_ADR21Q3.xml',
        'dataset/xml_download/faers_xml_2021Q3/XML/2_ADR21Q3.xml',
        'dataset/xml_download/faers_xml_2021Q3/XML/3_ADR21Q3.xml'],
        # 2021 Q4
        ['dataset/xml_download/faers_xml_2021Q4/XML/1_ADR21Q4.xml',
        'dataset/xml_download/faers_xml_2021Q4/XML/2_ADR21Q4.xml',
        'dataset/xml_download/faers_xml_2021Q4/XML/3_ADR21Q4.xml'],
        # 2022 Q1
        ['dataset/xml_download/faers_xml_2022Q1/xml/1_ADR22Q1.xml',
        'dataset/xml_download/faers_xml_2022Q1/xml/2_ADR22Q1.xml',
        'dataset/xml_download/faers_xml_2022Q1/xml/3_ADR22Q1.xml'],
        # 2022 Q2
        ['dataset/xml_download/faers_xml_2022Q2/xml/1_ADR22Q2.xml',
        'dataset/xml_download/faers_xml_2022Q2/xml/2_ADR22Q2.xml',
        'dataset/xml_download/faers_xml_2022Q2/xml/3_ADR22Q2.xml'],
        # 2022 Q3
        ['dataset/xml_download/faers_xml_2022Q3/xml/1_ADR22Q3.xml', 
        'dataset/xml_download/faers_xml_2022Q3/xml/2_ADR22Q3.xml', 
        'dataset/xml_download/faers_xml_2022Q3/xml/3_ADR22Q3.xml'],
        # 2022 Q4
        ['dataset/xml_download/faers_xml_2022Q4/xml/1_ADR22Q4.xml',
        'dataset/xml_download/faers_xml_2022Q4/xml/2_ADR22Q4.xml',
        'dataset/xml_download/faers_xml_2022Q4/xml/3_ADR22Q4.xml'],
        # 2023 Q1
        ['dataset/xml_download/faers_xml_2023Q1/xml/1_ADR23Q1.xml',
         'dataset/xml_download/faers_xml_2023Q1/xml/2_ADR23Q1.xml',
         'dataset/xml_download/faers_xml_2023Q1/xml/3_ADR23Q1.xml'],
        # 2023 Q2
        ['dataset/xml_download/faers_xml_2023Q2/xml/1_ADR23Q2.xml',
         'dataset/xml_download/faers_xml_2023Q2/xml/2_ADR23Q2.xml',
         'dataset/xml_download/faers_xml_2023Q2/xml/3_ADR23Q2.xml'],
        # 2023 Q3
        ['dataset/xml_download/faers_xml_2023Q3/xml/1_ADR23Q3.xml',
         'dataset/xml_download/faers_xml_2023Q3/xml/2_ADR23Q3.xml',
         'dataset/xml_download/faers_xml_2023Q3/xml/3_ADR23Q3.xml'],
        # 2023 Q4
        ['dataset/xml_download/faers_xml_2023Q4/xml/1_ADR23Q4.xml',
         'dataset/xml_download/faers_xml_2023Q4/xml/2_ADR23Q4.xml',
         'dataset/xml_download/faers_xml_2023Q4/xml/3_ADR23Q4.xml'],
        # 2024 Q1
        ['dataset/xml_download/faers_xml_2024Q1/xml/1_ADR24Q1.xml',
         'dataset/xml_download/faers_xml_2024Q1/xml/2_ADR24Q1.xml',
         'dataset/xml_download/faers_xml_2024Q1/xml/3_ADR24Q1.xml'],
        # 2024 Q2
        ['dataset/xml_download/faers_xml_2024Q2/xml/1_ADR24Q2.xml',
         'dataset/xml_download/faers_xml_2024Q2/xml/2_ADR24Q2.xml',
         'dataset/xml_download/faers_xml_2024Q2/xml/3_ADR24Q2.xml'],
        # 2024 Q3
        ['dataset/xml_download/faers_xml_2024Q3/xml/1_ADR24Q3.xml',
         'dataset/xml_download/faers_xml_2024Q3/xml/2_ADR24Q3.xml',
         'dataset/xml_download/faers_xml_2024Q3/xml/3_ADR24Q3.xml'],
    ]

        
    # Process each batch of XML files
    for batch in xml_batches:
        quarter_name = get_quarter_name(batch[0])
        print(f'\nProcessing {quarter_name}...')
        process_quarter(batch, quarter_name)
        
        # Force garbage collection after processing each quarter
        gc.collect()
        print(f'Memory cleaned after processing {quarter_name}')
    
    
    # Generate condition123 subsets for years of interest
    # generate_condition123_subsets(2023,2024)
    
    # Generate condition123 subsets for all years 
    generate_condition123_subsets("all")
    
    count_condition123_for_all_csv_files(directory_path= r"dataset/condition123subsets", output_path = r"dataset/condition123subsets/statistics/summary.csv")
    
    
    performStandardizedDrugName2DrugBankID(base_dir = 'dataset/condition123subsets', drugbank_csv_file = 'dataset/dictionary/Drugbank_ID_SMILE_all_structure links.csv', drugbank_xml_file = 'dataset/dictionary/DrugBankFullDataBase.xml')
    
    # Use SapBERT to generate edge embeddings for side effect names
    get_se_unique(file_path=r'dataset\dictionary\Side_effects_unique.csv')
    
    generate_sapbert_embedding_for_se(input_path=r'dataset\dictionary\Side_effects_unique.csv', output_path=r'dataset\dictionary\Side_effects_unique.csv', tokenizer_path="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",model_path="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    
    # Move csv files containing "none" values to none folder
    os.makedirs(os.path.join("dataset/condition123subsets", "none"), exist_ok=True)

    for file in os.listdir("dataset/condition123subsets"):
        if file.endswith(".csv"):
            if "2012" in file or "2013" in file or "2014Q1" in file or "2014Q2" in file:
                shutil.move(os.path.join("dataset/condition123subsets", file), os.path.join("dataset/condition123subsets", "none", file))
    
    generate_se_dict_for_quarters()
    count_unique_se_and_cui()
    
    add_stratified_se_to_all_csv_files()
    count_stratified_se_in_given_directory()
    
    generate_positive_samples_for_each_quarter()
    
    merge_and_deduplicate_csv(input_path=r'dataset\condition123subsets', output_file = r'dataset\condition123subsets\merged_deduplicated_condition123_2014Q3_2024Q3_step3.csv')
    
    clean_DrugBankID()
    
    delete_supersets_in_evaluation_subset()
    
    run_generate_negative_samples()
    
    # Deduplicate negative samples
    input_file = r"dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv"
    output_file = r"dataset\condition123subsets\negative_sample_statistics.txt"

    try:
        deduplicate_negative_samples(input_file, output_file)
        print("Deduplication completed!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
    check_if_negative_sample_happen_to_be_positive_sample()
    
    
    
    # Add time column in negative samples 
    append_time_column()
    
    # Split positive samples by quarter
    positive_samples_input_path = r'dataset\condition123subsets\positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'
    split_csv_by_time(input_csv_path=positive_samples_input_path)

    # Split negative samples by quarter
    negative_samples_input_path = r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'
    split_csv_by_time(input_csv_path=negative_samples_input_path)
    
    # Clean up positive and negative samples in HGNN dataset
    modify_csv_files('dataset\\evaluation_subset')