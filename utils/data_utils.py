# Import libraries
import os
import re
import gc
import ast
import glob
import shutil
import torch
import pickle
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm
from zipfile_deflate64 import ZIP_DEFLATED64, ZipFile
import xml.etree.ElementTree as ET
from datetime import date, datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from collections import defaultdict


def rename_and_extract_zip(base_path):
    """
    Process zip files in a given base path by renaming them and extracting the contents
    
    Parameters
    ----------
    base_path : str
        The base path to look for zip files in
    """
    
    if not os.path.exists(base_path):
        raise Exception(f"Path {base_path} does not exist")
    
    zip_files = [f for f in os.listdir(base_path) if f.endswith('.zip')]
    
    for zip_file in tqdm(zip_files, desc="Processing zip files"):
        old_path = os.path.join(base_path, zip_file)
        new_name = zip_file.replace('q', 'Q').replace('Q', 'Q')
        new_path = os.path.join(base_path, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            zip_file = new_name
        
        folder_name = os.path.splitext(zip_file)[0]
        extract_path = os.path.join(base_path, folder_name)
        
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        os.makedirs(extract_path)
        
        with ZipFile(os.path.join(base_path, zip_file), 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc=f"Extracting {zip_file}", leave=False):
                zip_ref.extract(file, extract_path)
                
            

def date_normalize(formate, dat):
    """Normalize dates to days since 2000-01-01"""
    stand_date = date(2000, 1, 1)
    try:
        if formate == '102':  # yyyymmdd
            current_date = date(int(dat[:4]), int(dat[4:6]), int(dat[6:8]))
        elif formate == '610':  # yyyymm
            current_date = date(int(dat[:4]), int(dat[4:6]), 1)
        elif formate == '602':  # yyyy
            current_date = date(int(dat[:4]), 1, 1)
        delta = current_date - stand_date
        return delta.days
    except:
        return '0'

def process_xml_files(xml_files):
    """Process multiple XML files and combine them"""
    root = None
    for xml_file in xml_files:
        print(f'Processing {xml_file}')
        try:
            data = ET.parse(xml_file).getroot()
            if root is None:
                root = data
            else:
                root.extend(data)
                print(f'Finished merging {xml_file}')
        except FileNotFoundError:
            print(f'Error: File not found - {xml_file}')
            raise
        except ET.ParseError:
            print(f'Error: Invalid XML file - {xml_file}')
            raise
    return root

def process_xml_files_2012Q4(xml_files):
    
    """
    Process multiple XML files and combine them. This function is used to process
    quarters newer than 2012Q4. It uses the latin1 encoding to parse the XML files.

    Parameters
    ----------
    xml_files : list
        List of paths to XML files to process

    Returns
    -------
    root : ElementTree
        The root of the combined XML tree
    """
    root = None
    for xml_file in xml_files:
        print(f'Processing {xml_file}')
        try:
            parser = ET.XMLParser(encoding="latin1")  
            data = ET.parse(xml_file, parser=parser)  
            if root is None:
                root = data.getroot()  
            else:
                root.extend(data.getroot())
                print(f'Finished merging {xml_file}')
        except FileNotFoundError:
            print(f'Error: File not found - {xml_file}')
            raise
        except ET.ParseError:
            print(f'Error: Invalid XML file - {xml_file}')
            raise
    return root

def process_quarter(xml_files, quarter_name):
    """Process XML files for a specific quarter"""
    # Create output directory if it doesn't exist
    output_dir = 'dataset/pk'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created output directory: {output_dir}')
    
    # Process XML files
    
    # For dataset newer than 2012Q4, use this line
    root = process_xml_files(xml_files)
    
    # For dataset on 2012Q4, use this line
    # root = process_xml_files_2012Q4(xml_files)
    nmb_reports = len(root)
    print(f'Total number of reports for {quarter_name}: {nmb_reports}')

    # Initialize counters
    count = 0
    miss_admin = miss_patient = miss_reaction = miss_drug = 0
    dic = {}

    # Process each safety report
    for report in tqdm(root.findall('safetyreport')):
        try:
            # Administrative Information
            try:
                version = report.find('safetyreportversion').text
            except:
                version = '1'
            
            report_id = report.find('safetyreportid').text
            
            try:
                case_id = report.find('companynumb').text
            except:
                case_id = '0'
            
            try:
                country = report.find('primarysource')[0].text
            except:
                country = 'unknown'
            
            if country == 'COUNTRY NOT SPECIFIED':
                country = 'unknown'
            
            try:
                qualify = report.find('primarysource')[1].text
            except:
                qualify = '6'
            
            if qualify not in {'1', '2', '3', '4', '5', '6', '7'}:
                qualify = '0'
            
            try:
                serious = report.find('serious').text
            except:
                serious = '-1'
            
            # Process serious subtypes
            serious_subtype = []
            for subtype in ['seriousnessdeath', 'seriousnesslifethreatening', 
                          'seriousnesshospitalization', 'seriousnessdisabling',
                          'seriousnesscongenitalanomali', 'seriousnessother']:
                try:
                    value = report.find(subtype).text
                except:
                    value = '0'
                serious_subtype.append(value)
            
        except:
            miss_admin += 1
            continue

        # Process dates
        try:
            receivedateformat = report.find('receivedateformat').text
            receivedate = report.find('receivedate').text
            receivedate = date_normalize(receivedateformat, receivedate)
        except:
            receivedate = '0'
        
        try:
            receiptdateformat = report.find('receiptdateformat').text
            receiptdate = report.find('receiptdate').text
            receiptdate = date_normalize(receiptdateformat, receiptdate)
        except:
            receiptdate = '0'

        # Process patient information
        for patient in report.findall('patient'):
            # Demographic Information
            try:
                age = patient.find('patientonsetage').text
                ageunit = patient.find('patientonsetageunit').text
            except:
                age = -1
                ageunit = '801'

            # Normalize age
            try:
                age = int(age)
                if age != -1:
                    if ageunit == '800':  # Decade
                        age = '-1'
                    elif ageunit == '801':  # Year
                        pass
                    elif ageunit == '802':  # Month
                        age = int(age/12)
                    elif ageunit == '803':  # Week
                        age = int(age/52)
                    elif ageunit == '804':  # Day
                        age = int(age/365)
                    elif ageunit == '805':  # Hour
                        age = int(age/(24*365))
            except:
                age = -1

            try:
                gender = patient.find('patientsex').text
            except:
                gender = '0'
            
            try:
                weight = patient.find('patientweight').text
            except:
                weight = '0'

            # Process reactions
            reaction_list = []
            for side_ in patient.findall('reaction'):
                try:
                    try:
                        PT_code = side_[0].text
                    except:
                        PT_code = '0'
                    try:
                        PT = side_[1].text
                    except:
                        PT = 'none'
                    try:
                        outcome = side_[2].text
                    except:
                        outcome = '6'
                    reaction = [PT_code, PT, outcome]
                except:
                    continue
                reaction_list.append(reaction)
            
            if not reaction_list:
                miss_reaction += 1
                continue

            # Process drugs
            drug_list = []
            for drug_ in patient.findall('drug'):
                try:
                    try:
                        char = drug_.find('drugcharacterization').text
                    except:
                        char = '0'
                    
                    try:
                        product = drug_.find('medicinalproduct').text
                    except:
                        product = 'none'

                    # Process drug dosage information
                    try:
                        dorse = drug_.find('drugstructuredosagenumb').text
                        unit = drug_.find('drugstructuredosageunit').text
                        drugseparatedosagenumb = drug_.find('drugseparatedosagenumb').text
                        drugintervaldosageunitnumb = drug_.find('drugintervaldosageunitnumb').text
                        drugintervaldosagedefinition = drug_.find('drugintervaldosagedefinition').text
                        form = drug_.find('drugdosageform').text
                    except:
                        dorse = unit = drugseparatedosagenumb = drugintervaldosageunitnumb = \
                        drugintervaldosagedefinition = form = '0'

                    try:
                        route = drug_.find('drugadministrationroute').text
                        if route == '048':
                            route = '1'  # oral
                        elif route == '061':
                            route = '2'  # Topical
                    except:
                        route = '0'

                    try:
                        indication = drug_.find('drugindication').text
                    except:
                        indication = 'none'

                    # Process drug dates
                    try:
                        start_format = drug_.find('drugstartdateformat').text
                        start_date = drug_.find('drugstartdate').text
                        start_date = date_normalize(start_format, start_date)
                    except:
                        start_date = '0'

                    try:
                        end_format = drug_.find('drugenddateformat').text
                        end_date = drug_.find('drugenddate').text
                        end_date = date_normalize(end_format, end_date)
                    except:
                        end_date = receiptdate if receiptdate != '0' else '0'

                    try:
                        action = drug_.find('actiondrug').text
                    except:
                        action = '5'

                    try:
                        additional = drug_.find('drugadditional').text
                    except:
                        additional = '3'

                    try:
                        readm = drug_.find('drugrecurreadministration').text
                    except:
                        readm = '3'

                    try:
                        substance = drug_.find('activesubstance')[0].text
                    except:
                        substance = 'none'

                    drug = [char, product, dorse, unit, drugseparatedosagenumb,
                           drugintervaldosageunitnumb, drugintervaldosagedefinition,
                           form, route, indication, start_date, end_date,
                           action, readm, additional, substance]
                    drug_list.append(drug)
                except:
                    continue

            if not drug_list:
                miss_drug += 1
                continue

            # Store the processed report
            dic[count] = [version, report_id, case_id, country, qualify, serious,
                         *serious_subtype,
                         receivedate, receiptdate,
                         age, gender, weight, reaction_list, drug_list]
            count += 1

    # Save processed data
    output_path = os.path.join(output_dir, f'{quarter_name}.pk')
    pickle.dump(dic, open(output_path, 'wb'))
    
    print(f'Processing complete for {quarter_name}. Total reports processed: {count}')
    print(f'Missing reports - Admin: {miss_admin}, Patient: {miss_patient}, '
          f'Reaction: {miss_reaction}, Drug: {miss_drug}')

def get_quarter_name(xml_path):
    """Extract quarter name from XML path
    Example: from 'dataset/xml_download/faers_xml_2023Q1/XML/1_ADR23Q1.xml'
    extracts '2023Q1'
    """
    try:
        # Split path by '/' and find the part containing 'faers_xml'
        for part in xml_path.split('/'):
            if 'faers_xml_' in part:
                # Extract the quarter part (e.g., '2023Q1' from 'faers_xml_2023Q1')
                quarter = part.split('faers_xml_')[1]
                return quarter
        raise ValueError(f'Could not find quarter information in path: {xml_path}')
    except Exception as e:
        raise ValueError(f'Error extracting quarter from path {xml_path}: {str(e)}')
    



def extract_se_names(se_entry):
    """Extract side effect names from SE column and deduplicate"""
    if se_entry is None or (isinstance(se_entry, float) and np.isnan(se_entry)):
        return []
    
    try:
        if isinstance(se_entry, str):
            se_entry = ast.literal_eval(se_entry)
            
        if not isinstance(se_entry, list):
            return []
        
        # Extract side effect names (second element of each sublist)
        se_names = [item[1] for item in se_entry if isinstance(item, list) and len(item) > 2]
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(se_names))
        
    except Exception as e:
        print(f"Error in extract_se_names: {str(e)}")
        return []

def determine_condition(row):
    """Determine condition based on number_of_1, number_of_2, and number_of_3"""
    try:
        n1 = row['number_of_1']
        n2 = row['number_of_2']
        n3 = row['number_of_3']
        
        if n1 == 0 and n2 == 0 and n3 >= 2:
            return 1
        elif n2 == 0 and n3 >= 1 and n1 >= 1:
            return 2
        elif n1 >= 1 and n2 >= 1 and n3 >= 1:
            return 3
        else:
            return 'NA'
            
    except Exception as e:
        print(f"Error in determine_condition: {str(e)}")
        return 'NA'

def extract_drug_info(drugs):
    """Extract drug's role and standardized name"""
    if drugs is None or (isinstance(drugs, float) and np.isnan(drugs)):
        return []
        
    try:
        if not isinstance(drugs, list):
            return []
        
        # Extract role (first element) and standardized name (last element)
        drug_info = [[item[0], item[-1]] for item in drugs if isinstance(item, list) and len(item) > 1]
        
        return drug_info
        
    except Exception as e:
        print(f"Error in extract_drug_info: {str(e)}")
        return []

def process_drug_list(entry):
    """Process drug list and deduplicate entries"""
    if entry is None or (isinstance(entry, float) and np.isnan(entry)):
        return []
        
    try:
        if not isinstance(entry, list):
            return []
        
        unique_drugs = {}
        
        for item in entry:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
                
            try:
                role, drug = item[0], item[1]
                drug = str(drug).strip()
                
                if '\\' in drug:
                    parts = drug.split('\\')
                    for part in parts:
                        part = part.strip()
                        if part and part not in unique_drugs:
                            unique_drugs[part] = str(role)
                else:
                    if drug and drug not in unique_drugs:
                        unique_drugs[drug] = str(role)
            except Exception as e:
                print(f"Error processing drug item: {str(e)}")
                continue
        
        return [[role, drug] for drug, role in unique_drugs.items()]
        
    except Exception as e:
        print(f"Error in process_drug_list: {str(e)}")
        return []

def count_role_numbers(drug_list):
    """Count the number of drugs by role"""
    try:
        if not isinstance(drug_list, list):
            return {
                'total_number_of_drugs': 0,
                'number_of_1': 0,
                'number_of_2': 0,
                'number_of_3': 0
            }
            
        role_counts = {
            'total_number_of_drugs': len(drug_list),
            'number_of_1': 0,
            'number_of_2': 0,
            'number_of_3': 0
        }
        
        for drug in drug_list:
            try:
                role = str(drug[0])
                if role == '1':
                    role_counts['number_of_1'] += 1
                elif role == '2':
                    role_counts['number_of_2'] += 1
                elif role == '3':
                    role_counts['number_of_3'] += 1
            except (IndexError, TypeError):
                continue
        
        return role_counts
    except Exception as e:
        print(f"Error in count_role_numbers: {str(e)}")
        return {
            'total_number_of_drugs': 0,
            'number_of_1': 0,
            'number_of_2': 0,
            'number_of_3': 0
        }

def generate_statistics(df):
    """Generate statistical analysis for numerical columns"""
    columns_to_analyze = ['total_number_of_drugs', 'number_of_1', 'number_of_2', 'number_of_3']
    stats_dict = {
        'metric': ['Maximum', 'Maximum Row Index', 'Minimum', 'Mean', 'Median', '25th Percentile', '75th Percentile']
    }
    
    text_output = []
    text_output.append("Statistical Analysis Report")
    text_output.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for column in columns_to_analyze:
        try:
            max_index = df[column].idxmax()
            
            stats = [
                df[column].max(),
                max_index,
                df[column].min(),
                round(df[column].mean(), 4),
                df[column].median(),
                round(df[column].quantile(0.25), 4),
                round(df[column].quantile(0.75), 4)
            ]
            
            stats_dict[column] = stats
            
            text_output.append(f"\nDetailed Analysis for Column: {column}")
            text_output.append(f"Maximum value: {df[column].max()}")
            text_output.append(f"Row index: {max_index}")
            text_output.append("-" * 50)
        except Exception as e:
            print(f"Error analyzing column {column}: {str(e)}")
            continue
    
    return pd.DataFrame(stats_dict), '\n'.join(text_output)


def split_and_deduplicate_drug_info(drug_info):
    """
    Split drug names containing '\\' and deduplicate the resulting list
    """
    if not isinstance(drug_info, list):
        return []
    
    result = []
    seen = set()  # for deduplication
    
    for item in drug_info:
        if not isinstance(item, list) or len(item) != 2:
            continue
            
        role, drug_name = item
        
        # Check if the string contains a backslash
        if '\\' in drug_name:
            # Split drug names containing '\\'
            drug_names = drug_name.split('\\')
            for single_drug in drug_names:
                single_drug = single_drug.strip()
                # Create a tuple to check for duplicates
                item_tuple = (role, single_drug)
                if item_tuple not in seen:
                    seen.add(item_tuple)
                    result.append([role, single_drug])
        else:
            # If the string doesn't contain a backslash, add it as is
            item_tuple = (role, drug_name)
            if item_tuple not in seen:
                seen.add(item_tuple)
                result.append([role, drug_name])
    
    return result


def generate_condition123_subsets(*years):
    '''
    Generate subsets for condition 1, 2, and 3
    
    Params:
        years: list of years
        
    Example:
        generate_condition123_subsets("all")
        generate_condition123_subsets(2023, 2024)
    '''
    try:
        # Create output directory
        os.makedirs('dataset/condition123subsets/statistics', exist_ok=True)
        os.makedirs('dataset/unfiltered_full_dataset', exist_ok=True)
        
        # Get list of PK files
        if years[0] == "all":
            pk_files = glob.glob('dataset/pk/*.pk')
        else:
            pk_files = []
            for year in years:
                pk_files.extend(glob.glob(f'dataset/pk/*{year}*.pk'))
    
        for pk_file in pk_files:
            try:
                # Extract quarter from file name (e.g., get "2024Q3" from "2024Q3.pk")
                quarter = os.path.basename(pk_file).split('.')[0]
                print(f"\nProcessing {quarter}...")
                
                print("Reading input data...")
                input_dict = pd.read_pickle(pk_file)
                
                print("Processing data...")
                rows = []
                total = len(input_dict)
                
                for idx, (case_id, value) in enumerate(input_dict.items()):
                    try:
                        drugs = value[-1]
                        se = value[-2]
                        report_id = value[1]
                        
                        # Process SE
                        se_names = extract_se_names(se)
                        
                        # Process drugs
                        drug_info = extract_drug_info(drugs)
                        drugs_role_and_name = [[item[0], item[1]] for item in drugs] if isinstance(drugs, list) else []
                        drugs_deduplicated = process_drug_list(drugs_role_and_name)
                        role_counts = count_role_numbers(drugs_deduplicated)
                        
                        # Convert drug_info to string for deduplication
                        drug_info_str = str(drug_info)
                        drug_info = ast.literal_eval(drug_info_str)
                        drug_info = [list(x) for x in dict(((tuple(i), i) for i in drug_info)).values()]
                        
                        # Split and deduplicate
                        drug_info = split_and_deduplicate_drug_info(drug_info)
                        
                        row_data = {
                            'case_id': case_id,
                            'SE': se,
                            'SE_name': se_names,
                            'report_id': report_id,
                            'drugs': drugs,
                            'StandardizedDrugName_DrugRole': drug_info,
                            'drugs_role_and_name_deduplicate': drugs_deduplicated,
                            'total_number_of_drugs': role_counts['total_number_of_drugs'],
                            'number_of_1': role_counts['number_of_1'],
                            'number_of_2': role_counts['number_of_2'],
                            'number_of_3': role_counts['number_of_3']
                        }
                        
                        rows.append(row_data)
                        
                        if (idx + 1) % 1000 == 0:
                            print(f"Processed {idx + 1}/{total} entries")
                            
                    except Exception as e:
                        print(f"Error processing case {case_id}: {str(e)}")
                        continue
                
                df = pd.DataFrame(rows)
                
                # Add condition column
                df['condition'] = df.apply(determine_condition, axis=1)
                
                # Output dataframe information
                print("\nShape of dataframe:", df.shape)
                print("\nColumns in dataframe:", df.columns.tolist())
                print("\nValue counts for condition column:")
                print(df['condition'].value_counts())
                
                # Convert 'condition' column to numeric
                df['condition'] = pd.to_numeric(df['condition'], errors='coerce')
                
                # Extract rows with condition 1,2,3
                condition_123_df = df[df['condition'].isin([1, 2, 3])]
                print(f"\nNumber of rows with condition 1,2,3: {len(condition_123_df)}")
                
                print("\nSaving main results...")
                
                # Save the full dataset
                df.to_pickle(f'dataset/unfiltered_full_dataset/FAERS_{quarter}_processed.pk')
                df.to_csv(f'dataset/unfiltered_full_dataset/FAERS_{quarter}_processed.csv', index=False, encoding='utf-8')
                
                # Generate and save statistics
                print("Generating statistics...")
                stats_df, detailed_analysis = generate_statistics(df)
                stats_df.to_csv(f'dataset/condition123subsets/statistics/drugs_statistics_summary_{quarter}.csv', index=False, encoding='utf-8')
                with open(f'dataset/condition123subsets/statistics/drugs_statistics_detailed_{quarter}.txt', 'w', encoding='utf-8') as f:
                    f.write(detailed_analysis)
                
                # Save subset for condition 1,2,3
                print("\nSaving condition 1,2,3 data...")
                selected_columns = ['report_id', 'SE_name', 'StandardizedDrugName_DrugRole', 'condition']
                final_df = condition_123_df[selected_columns]
                
                print(f"Shape of final_df: {final_df.shape}")
                print("First few rows of final_df:")
                print(final_df.head())
                
                # Save the final dataframe 
                final_df.to_csv(f'dataset/condition123subsets/condition123_subset_{quarter}.csv', 
                               index=False, 
                               encoding='utf-8')
                
                # Check if the files were saved successfully
                try:
                    saved_df = pd.read_csv(f'dataset/condition123subsets/condition123_subset_{quarter}.csv')
                    print(f"\nSuccessfully saved and verified condition123_subset_{quarter}.csv with {len(saved_df)} rows")
                except Exception as e:
                    print(f"\nError verifying saved file: {str(e)}")
                
                print(f"\nProcessing for {quarter} completed successfully!")
                print("Output files:")
                print(f"  - dataset/FAERS_{quarter}_processed.pk")
                print(f"  - dataset/FAERS_{quarter}_processed.csv") 
                print(f"  - dataset/drugs_statistics_summary_{quarter}.csv")
                print(f"  - dataset/drugs_statistics_detailed_{quarter}.txt")
                print(f"  - dataset/condition123subsets/condition123_subset_{quarter}.csv")
                
            except Exception as e:
                print(f"Error processing file {pk_file}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    


def count_condition123_in_one_csv_file(file_path):
    """
    Count the number of rows for condition 1,2,3 in one CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    dict: Contains counts for each condition value
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Count rows for each condition value
        condition_1_count = len(df[df['condition'] == 1])
        condition_2_count = len(df[df['condition'] == 2])
        condition_3_count = len(df[df['condition'] == 3])
        total_conditions = len(df[df['condition'].isin([1, 2, 3])])
        
        return {
            'file_name': os.path.basename(file_path),
            'condition_1': condition_1_count,
            'condition_2': condition_2_count,
            'condition_3': condition_3_count,
            'total': total_conditions
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def count_condition123_for_all_csv_files(directory_path, output_path):
    """
    Count the number of rows for condition 1,2,3 in all CSV files in a given directory
    
    Parameters:
    directory_path (str): Path to the directory to process
    output_path (str): Path where to save the statistics CSV file
    
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
        return
    
    # Store results for all files
    all_results = []
    
    # Process all files in directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            result = count_condition123_in_one_csv_file(file_path)
            if result:
                all_results.append(result)
    
    # Print statistics
    print("\nStatistics:")
    print("-" * 80)
    print(f"{'File Name':<30} {'Condition=1':<12} {'Condition=2':<12} {'Condition=3':<12} {'Total':<12}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['file_name']:<30} {result['condition_1']:<12} {result['condition_2']:<12} "
              f"{result['condition_3']:<12} {result['total']:<12}")
    
    # Calculate totals for all files
    total_condition_1 = sum(r['condition_1'] for r in all_results)
    total_condition_2 = sum(r['condition_2'] for r in all_results)
    total_condition_3 = sum(r['condition_3'] for r in all_results)
    grand_total = sum(r['total'] for r in all_results)
    
    print("-" * 80)
    print(f"{'Total':<30} {total_condition_1:<12} {total_condition_2:<12} "
          f"{total_condition_3:<12} {grand_total:<12}")
    
    # Create a DataFrame from results and save to CSV
    results_df = pd.DataFrame(all_results)
    # Add a row for totals
    totals_row = pd.DataFrame([{
        'file_name': 'Total',
        'condition_1': total_condition_1,
        'condition_2': total_condition_2,
        'condition_3': total_condition_3,
        'total': grand_total
    }])
    results_df = pd.concat([results_df, totals_row], ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"\nStatistics saved to: {output_path}")
    
    


def rename_files_step1(directory):
    """
    Rename files in a directory that do not end with '_step1.csv'
    If the target file already exists, delete it first
    """
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            if not filename.endswith('_step1.csv'):
                old_path = os.path.join(directory, filename)
                new_filename = filename.replace('.csv', '_step1.csv')
                new_path = os.path.join(directory, new_filename)
                
                # If the target file already exists, delete it first
                if os.path.exists(new_path):
                    os.remove(new_path)
                    print(f"Removed existing file: {new_filename}")
                
                os.rename(old_path, new_path)
                print(f"Renamed {filename} to {new_filename}")


def clean_drug_name(name):
    """
    1. Convert a drug name to uppercase and remove whitespace
    2. Remove common drug salt forms and hydrate suffixes (e.g., hydrochloride/HCL, sulfate, etc.), keeping only the main drug component name
    """
    
    # If name is not a string, return it as is
    if not isinstance(name, str):
        return name
    
    # Convert to uppercase and remove leading/trailing whitespace
    name = name.upper().strip()
    
    # Remove salt and hydrate suffixes
    salt_forms = [
        'HYDROCHLORIDE', 'HCL', 'HYDROBROMIDE', 'HBR',
        'SULFATE', 'SULPHATE',
        'SODIUM', 'NA',
        'POTASSIUM', 'K',
        'CALCIUM', 'CA',
        'MESYLATE', 'MESILATE',
        'MALEATE',
        'TARTRATE',
        'CITRATE',
        'PHOSPHATE',
        'ACETATE', 'AC',
        'SUCCINATE',
        'FUMARATE',
        'BROMIDE',
        'DIPROPIONATE',
        'MONOHYDRATE',
        'BESYLATE',
        'DECANOATE',
        'AXETIL',
        'ETHANOLATE',
        'BITARTRATE',
        'PROPANEDIOL',
        'MAGNESIUM', 'MG',
        'DISOPROXIL',
        'TRIFENATATE',
        'HYCLATE'
    ]
    
    # Split the name into individual words using spaces
    words = name.split()
    # Keep words that are not in the salt forms list
    cleaned_words = [word for word in words if word not in salt_forms]
    # Join the cleaned words back into a single string
    cleaned_name = ' '.join(cleaned_words)
    # Return the cleaned drug name
    return cleaned_name.strip()

def create_drugname_to_id_mapping_from_xml(xml_file):
    """
    Create a mapping from drug names to DrugBankID
    """
    print("Creating drug name to ID mapping from XML...")
    name_to_id = {}
    ns = {'db': 'http://www.drugbank.ca'}
    context = ET.iterparse(xml_file, events=('end',))
    
    for event, elem in tqdm(context):
        if elem.tag == '{http://www.drugbank.ca}drug':
            # Get DrugBank ID
            primary_id = elem.find('.//db:drugbank-id[@primary="true"]', ns)
            if primary_id is not None:
                drug_id = primary_id.text
                
                # Store all DrugBankIDs
                db_ids = elem.findall('.//db:drugbank-id', ns)
                for db_id in db_ids:
                    if db_id.text:
                        name_to_id[db_id.text.strip().upper()] = drug_id
                        # Store all DrugBankIDs and their cleaned versions
                        clean_id = clean_drug_name(db_id.text)
                        if clean_id:
                            name_to_id[clean_id] = drug_id
                
                # Store the primary name
                name_elem = elem.find('.//db:name', ns)
                if name_elem is not None and name_elem.text:
                    name = name_elem.text.strip().upper()
                    name_to_id[name] = drug_id
                    # Store all DrugBankIDs and their cleaned versions
                    clean_name = clean_drug_name(name)
                    if clean_name:
                        name_to_id[clean_name] = drug_id
                
                # Store synonyms
                synonyms = elem.findall('.//db:synonyms/db:synonym', ns)
                for syn in synonyms:
                    if syn.text:
                        syn_name = syn.text.strip().upper()
                        name_to_id[syn_name] = drug_id
                        # Store all DrugBankIDs and their cleaned versions
                        clean_syn = clean_drug_name(syn_name)
                        if clean_syn:
                            name_to_id[clean_syn] = drug_id
                
                # Store international brands
                brands = elem.findall('.//db:international-brands/db:international-brand/db:name', ns)
                for brand in brands:
                    if brand.text:
                        brand_name = brand.text.strip().upper()
                        name_to_id[brand_name] = drug_id
                        # Store all DrugBankIDs and their cleaned versions
                        clean_brand = clean_drug_name(brand_name)
                        if clean_brand:
                            name_to_id[clean_brand] = drug_id
                
                # Store products
                products = elem.findall('.//db:products/db:product/db:name', ns)
                for product in products:
                    if product.text:
                        prod_name = product.text.strip().upper()
                        name_to_id[prod_name] = drug_id
                        # Store all DrugBankIDs and their cleaned versions
                        clean_prod = clean_drug_name(prod_name)
                        if clean_prod:
                            name_to_id[clean_prod] = drug_id
            
            # Release memory
            elem.clear()
    
    return name_to_id

def StandardizedDrugName2DrugBankID(input_file, drugbank_csv_file, drugbank_xml_file, output_file):
    """
    Convert drug names to DrugBankIDs

    Args:
    input_file (str): Path to input CSV containing drug names and roles
    drugbank_csv_file (str): Path to DrugBank CSV mapping file
    drugbank_xml_file (str): Path to DrugBank XML mapping file  
    output_file (str): Path to save processed CSV

    Returns:
    None. 
    
    Saves processed DataFrame to output_file with columns:
    - report_id: Original report identifier
    - condition: Medical condition
    - DrugBankID_DrugRole: List of [drug_role, drugbank_id] pairs
    """
    # Read input file
    df = pd.read_csv(input_file)
    
    # Create a mapping from drug names to DrugBank IDs
    drugbank_df = pd.read_csv(drugbank_csv_file)
    csv_name_to_id = dict(zip(drugbank_df['Name'].str.upper(), drugbank_df['DrugBank ID']))
    
    # Create a mapping from drug names to DrugBank IDs
    xml_name_to_id = create_drugname_to_id_mapping_from_xml(drugbank_xml_file)
    
    # Merge the mappings. XML mapping has priority
    combined_mapping = {**csv_name_to_id, **xml_name_to_id}
    
    # Convert drug names to DrugBankID
    def convert_drug_names(row):
        try:
            drug_list = ast.literal_eval(row)
            converted = []
            unmatched = []
            
            for item in drug_list:
                drug_role = item[0]
                original_name = item[1]
                
                if original_name.upper() in ['NONE', 'NOS', 'UNSPECIFIED INGREDIENT', 'DIETARY SUPPLEMENT', 'HERBALS']:
                    converted.append([drug_role, original_name])
                    continue
                
                # Try direct match
                drug_id = combined_mapping.get(original_name.upper())
                
                # If direct match fails, try cleaning the name
                if drug_id is None:
                    clean_name = clean_drug_name(original_name)
                    drug_id = combined_mapping.get(clean_name)
                
                # If still no match, try fuzzy matching
                if drug_id is None:
                    clean_name_upper = clean_name.upper()
                    for db_name, db_id in combined_mapping.items():
                        db_name_upper = db_name.upper()
                        if clean_name_upper in db_name_upper or db_name_upper in clean_name_upper:
                            drug_id = db_id
                            break
                
                # If still no match, add to unmatched
                if drug_id is None:
                    unmatched.append(original_name)
                    drug_id = original_name
                
                converted.append([drug_role, drug_id])
            
            if unmatched:
                print(f"Unmatched drugs: {unmatched}")
            
            return str(converted)
        except Exception as e:
            print(f"Error processing row: {row}")
            print(f"Error message: {str(e)}")
            return row
    
    print("Converting drug names to DrugBank IDs...")
    df['DrugBankID_DrugRole'] = df['StandardizedDrugName_DrugRole'].apply(convert_drug_names)
    
    # Save the processed DataFrame and add a warning message if the file already exists
    if os.path.exists(output_file):
        print(f"Warning: Overwriting existing file {output_file}")
    result_df = df[['report_id', 'condition', 'DrugBankID_DrugRole']]
    result_df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")


def performStandardizedDrugName2DrugBankID(base_dir: str, drugbank_csv_file: str, drugbank_xml_file: str):
    """
    Perform the following steps to convert standardized drug names to DrugBank IDs:

    1. Rename all CSV files in the given base directory by appending '_step1' to the end of their names.
    2. Process each CSV file (with the new name) by converting the drug names to DrugBank IDs using the given DrugBank CSV and XML files.
    3. Save the processed data to a new CSV file with the same name as the input file but with '_step2' appended to the end.

    Args:
        base_dir (str): The base directory containing the CSV files to process.
        drugbank_csv_file (str): The path to the DrugBank CSV file containing the drug names and IDs.
        drugbank_xml_file (str): The path to the DrugBank XML file containing the drug names and IDs.

    Returns:
        None
    """

    
    # Rename all CSV files in the given base directory by appending '_step1' to the end of their names.
    rename_files_step1(base_dir)
    
    # Process each CSV file ending with '_step1.csv'
    for filename in os.listdir(base_dir):
        if filename.endswith('_step1.csv'):
            input_file = os.path.join(base_dir, filename)
            # Generate the output file name of step2
            output_filename = filename.replace('_step1.csv', '_step2.csv')
            output_file = os.path.join(base_dir, output_filename)
            
            print(f"Processing file: {filename}")
            # Convert standardized drug names to DrugBank IDs
            StandardizedDrugName2DrugBankID(input_file, drugbank_csv_file, drugbank_xml_file, output_file)
            print(f"Completed processing: {filename}")



def get_se_unique(file_path):
    """
    Process a CSV file to remove rows with NA values and duplicates based on 'umls_cui_from_meddra' column.
    
    Parameters:
    file_path (str): Path to the CSV file to be processed.
    
    This function performs the following steps:
    1. Reads the CSV file and prints the original number of rows.
    2. Removes rows with NA values in the 'umls_cui_from_meddra' column and prints the number of rows removed.
    3. Removes duplicate rows based on the 'umls_cui_from_meddra' column and prints the number of duplicates removed.
    4. Saves the processed DataFrame back to the original CSV file.
    5. Prints a summary of the processing steps.
    """

    print(f"Reading csv file: {file_path}")
    df = pd.read_csv(file_path)
    
    original_rows = len(df)
    print(f"Original number of rows: {original_rows}")
    
    # Remove NA in umls_cui_from_meddra column
    df_no_na = df.dropna(subset=['umls_cui_from_meddra'])
    rows_after_na_removal = len(df_no_na)
    rows_removed_na = original_rows - rows_after_na_removal
    print(f"Number of rows after NA removal: {rows_after_na_removal}")
    print(f"Number of deleted rows: {rows_removed_na}")
    
    # Deduplicate based on umls_cui_from_meddra
    df_unique = df_no_na.drop_duplicates(subset=['umls_cui_from_meddra'])
    final_rows = len(df_unique)
    rows_removed_duplicates = rows_after_na_removal - final_rows
    print(f"Number of final rows: {final_rows}")
    print(f"Number of deleted duplicate rows: {rows_removed_duplicates}")
    
    # Save the processed DataFrame back to the original CSV file
    df_unique.to_csv(file_path, index=False)
    print(f"File saved: {file_path}")
    

    print("\nSummary:")
    print(f"{'original_rows'} rows were processed")
    print(f"{'rows_removed_na'} rows with NA values were removed")
    print(f"{'rows_removed_duplicates'} deduplicated rows removed")
    print(f"{'final_rows'} rows were kept")



def generate_sapbert_embedding_for_se(input_path, output_path, tokenizer_path, model_path):
    """
    Generate SapBERT embeddings for side effect names in a CSV file.

    This function reads side effect names from an input CSV file, generates embeddings
    using a pre-trained SapBERT model, and saves the results to an output CSV file.

    Parameters:
        input_path (str): Path to the input CSV file containing the side effect names.
        output_path (str): Path to save the output CSV file with generated embeddings.
        tokenizer_path (str): Path to the tokenizer for the pre-trained SapBERT model.
        model_path (str): Path to the pre-trained SapBERT model.

    Returns:
        None

    The output CSV will contain the following columns:
    - umls_cui_from_meddra: The UMLS CUI identifiers from the original input.
    - side_effect_name: The original side effect names.
    - 768 columns representing the generated embeddings for each side effect name.
    """

    # Read the input CSV
    df = pd.read_csv(input_path)

    # Initialize SapBERT
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path).cuda()

    # Get side effect names
    all_names = df['side_effect_name'].tolist()

    # Generate embeddings using SapBERT
    bs = 128
    all_embs = []

    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i+bs],
                                        padding="max_length",
                                        max_length=50,
                                        truncation=True,
                                        return_tensors="pt")
        toks_cuda = {k: v.cuda() for k, v in toks.items()}
        with torch.no_grad():
            cls_rep = model(**toks_cuda)[0][:,0,:]
        all_embs.append(cls_rep.cpu().detach().numpy())

    # Concatenate all embeddings
    all_embs = np.concatenate(all_embs, axis=0)

    # Append the embeddings in 768 columns to the original DataFrame 
    embedding_df = pd.DataFrame(all_embs)  

    # Add the original columns
    result_df = pd.concat([
        df[['umls_cui_from_meddra', 'side_effect_name']],  
        embedding_df  
    ], axis=1)

    result_df.to_csv(output_path, index=False)
    



def collect_se_names_from_files(input_dir):
    """Collect all unique SE names from all matching CSV files."""
    print("Collecting SE names from all files...")
    all_se_names = set()  # Using set for automatic deduplication
    
    # Process all matching files in the directory
    for filename in os.listdir(input_dir):
        if filename.startswith('condition123_') and filename.endswith('_step1.csv'):
            input_path = os.path.join(input_dir, filename)
            print(f"Reading file: {filename}")
            
            # Read the input CSV file
            df = pd.read_csv(input_path)
            
            # Process each row in the SE_name column
            for se_string in df['SE_name']:
                try:
                    # Convert string representation to actual list
                    se_list = ast.literal_eval(se_string)
                    all_se_names.update(se_list)  # Update set with new names
                except:
                    continue
    
    # Convert set to sorted list
    return sorted(list(all_se_names))

def generate_embeddings(unique_se_names, tokenizer, model):
    """Generate embeddings for the unique SE names."""
    print("Generating embeddings...")
    bs = 128
    all_embs = []
    
    for i in tqdm(np.arange(0, len(unique_se_names), bs)):
        toks = tokenizer.batch_encode_plus(unique_se_names[i:i+bs],
                                         padding="max_length",
                                         max_length=50,
                                         truncation=True,
                                         return_tensors="pt")
        toks_cuda = {k: v.cuda() for k, v in toks.items()}
        with torch.no_grad():
            cls_rep = model(**toks_cuda)[0][:,0,:]
        all_embs.append(cls_rep.cpu().detach().numpy())
    
    return np.concatenate(all_embs, axis=0)

def find_most_similar(unlabeled_embeddings, labeled_embeddings, labeled_cui, labeled_names):
    """Find most similar labeled side effects."""
    results = []
    total_embeddings = len(unlabeled_embeddings)
    
    print("Finding most similar labeled side effects...")
    for i, unlabeled_embedding in enumerate(unlabeled_embeddings):
        if i % 100 == 0:
            print(f"Processing side effect {i}/{total_embeddings}")
        similarities = cosine_similarity(unlabeled_embedding.reshape(1, -1), labeled_embeddings)[0]
        top_index = np.argmax(similarities)
        results.append([
            labeled_cui[top_index],
            labeled_names[top_index],
            similarities[top_index]
        ])
    return results

def generate_se_dict_for_quarters(input_dir = r'dataset\condition123subsets'):
    
    # Collect all unique SE names
    unique_se_names = collect_se_names_from_files(input_dir)
    print(f"Total unique SE names collected: {len(unique_se_names)}")
    
    # Initialize SapBERT
    print("Initializing SapBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
    
    # Generate embeddings for all unique SE names
    all_embs = generate_embeddings(unique_se_names, tokenizer, model)
    
    # Load labeled side effects for comparison
    print("Loading labeled side effects...")
    labeled_se_file = r'dataset\dictionary\Side_effects_unique.csv'
    labeled_df = pd.read_csv(labeled_se_file)
    labeled_cui = labeled_df['umls_cui_from_meddra']
    labeled_names = labeled_df['side_effect_name']
    labeled_embeddings = labeled_df.iloc[:, 2:].values
    
    # Process similarities
    matches = find_most_similar(all_embs, labeled_embeddings, labeled_cui, labeled_names)
    
    # Create final output DataFrame
    final_df = pd.DataFrame({
        'SE_name_2014Q3_2024Q3': unique_se_names,
        'recommended_SE_name': [match[1] for match in matches],
        'recommended_umls_cui_from_meddra': [match[0] for match in matches],
        'cosine_similarity': [match[2] for match in matches]
    })
    
    # Save final results
    output_file = r'dataset\dictionary\SE_similarity_2014Q3_2024Q3.csv'
    final_df.to_csv(output_file, index=False)
    print(f"Processing complete! Final file saved as: {output_file}")

    

def count_unique_se_and_cui(df_path=r'dataset\dictionary\SE_similarity_2014Q3_2024Q3.csv'):
    df = pd.read_csv(df_path)

    # Count unique SE names in the SE_name_2014Q3_2024Q3 column
    unique_se_names = len(df['SE_name_2014Q3_2024Q3'].unique())
    print(f"Number of unique SE names (2014Q3-2024Q3): {unique_se_names}")

    # Count unique recommended_SE_name in the recommended_SE_name column
    unique_recommended_names = len(df['recommended_SE_name'].unique())
    print(f"Number of unique recommended_SE_name: {unique_recommended_names}")

    unique_recommended_names = len(df['recommended_umls_cui_from_meddra'].unique())
    print(f"Number of unique recommended_umls_cui_from_meddra: {unique_recommended_names}")


 

def add_stratified_se_to_one_csv_file(step1_file, step2_file, df_similarity):
    '''
    Add 'SE_below_0.8', 'SE_between_0.8_and_0.9', and 'SE_above_0.9' columns to the dataset of one quarter
    
    The input file (step1_file) is a CSV file containing the following columns:
        - report_id
        - condition
        - DrugBankID_DrugRole
        - SE_name
    
    The output file (step2_file) will have the same columns as the input file, plus three additional columns:
        - SE_below_0.8: a list of UMLS CUIs for side effects with similarity < 0.8
        - SE_between_0.8_and_0.9: a list of UMLS CUIs for side effects with similarity >= 0.8 and < 0.9
        - SE_above_0.9: a list of UMLS CUIs for side effects with similarity >= 0.9
    '''
    df_step1 = pd.read_csv(step1_file)
    df_step2 = pd.read_csv(step2_file)

    # Initialize the three new columns
    df_step2['SE_below_0.8'] = [[] for _ in range(len(df_step2))]
    df_step2['SE_between_0.8_and_0.9'] = [[] for _ in range(len(df_step2))]
    df_step2['SE_above_0.9'] = [[] for _ in range(len(df_step2))]

    # Process each row in the input file
    for idx, row in tqdm(df_step1.iterrows(), total=len(df_step1), 
                        desc=f"Processing {os.path.basename(step1_file)}"):
        se_list = ast.literal_eval(row['SE_name'])
        
        # Initialize lists to store UMLS CUIs for each similarity range
        below_0_8 = []
        between_0_8_and_0_9 = []
        above_0_9 = []

        # Process each side effect in the list
        for se_name in se_list:
            match = df_similarity[df_similarity['SE_name_2014Q3_2024Q3'] == se_name]
            
            if not match.empty:
                # Get the similarity and UMLS CUI for the matched side effect
                similarity = match.iloc[0]['cosine_similarity']
                umls_cui = match.iloc[0]['recommended_umls_cui_from_meddra']
                
                # Determine which list to add the UMLS CUI to based on the similarity
                if similarity < 0.8:
                    below_0_8.append(umls_cui)
                elif 0.8 <= similarity < 0.9:
                    between_0_8_and_0_9.append(umls_cui)
                else:  # similarity >= 0.9
                    above_0_9.append(umls_cui)

        # Update the output DataFrame with the lists of UMLS CUIs
        df_step2.at[idx, 'SE_below_0.8'] = below_0_8
        df_step2.at[idx, 'SE_between_0.8_and_0.9'] = between_0_8_and_0_9
        df_step2.at[idx, 'SE_above_0.9'] = above_0_9

    # Save the updated DataFrame to the output file
    df_step2.to_csv(step2_file, index=False)
    print(f"File updated: '{step2_file}'")

def add_stratified_se_to_all_csv_files():
    se_similarity_file = r'dataset\dictionary\SE_similarity_2014Q3_2024Q3.csv'
    df_similarity = pd.read_csv(se_similarity_file)

    base_path = 'dataset/condition123subsets'
    
    for file in os.listdir(base_path):
        if 'step1.csv' in file:
            step1_file = os.path.join(base_path, file)
            step2_file = os.path.join(base_path, file.replace('step1', 'step2'))
            
            add_stratified_se_to_one_csv_file(step1_file, step2_file, df_similarity)



def count_stratified_se_in_a_csv_file(string_list):
    """
    Count the stratified SE in a CSV file

    Parameters:
    string_list (str): a string containing a list of elements

    Returns:
    int: the number of elements in the list

    Note: if the string is empty or None, return 0
    """
    try:
        if pd.isna(string_list):
            return 0
        cleaned_str = string_list.strip().replace('"', "'")
        actual_list = ast.literal_eval(cleaned_str)
        return len(actual_list)
    except Exception as e:
        print(f"Error processing string: {string_list}, Error: {str(e)}")
        return 0

def count_stratified_se_in_given_directory():
    input_path = r'dataset\condition123subsets'
    output_dir = r'dataset\condition123subsets\statistics'
    os.makedirs(output_dir, exist_ok=True)
    output_file = r'dataset\condition123subsets\statistics\num_stratified_se.csv'
    
    pattern = os.path.join(input_path, 'condition123*step2.csv')
    csv_files = glob.glob(pattern)
    
    results = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            file_name = os.path.basename(file_path)
            
            stats = {
                'file_name': file_name,
                'SE_below_0.8_count': df['SE_below_0.8'].apply(count_stratified_se_in_a_csv_file).sum(),
                'SE_between_0.8_and_0.9_count': df['SE_between_0.8_and_0.9'].apply(count_stratified_se_in_a_csv_file).sum(),
                'SE_above_0.9_count': df['SE_above_0.9'].apply(count_stratified_se_in_a_csv_file).sum()
            }
            
            results.append(stats)
            print(f"Processed: {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    if results:
        results_df = pd.DataFrame(results)
        
        sum_row = {
            'file_name': 'sum',
            'SE_below_0.8_count': results_df['SE_below_0.8_count'].sum(),
            'SE_between_0.8_and_0.9_count': results_df['SE_between_0.8_and_0.9_count'].sum(),
            'SE_above_0.9_count': results_df['SE_above_0.9_count'].sum()
        }
        
        results_df = pd.concat([results_df, pd.DataFrame([sum_row])], ignore_index=True)
        
        results_df.to_csv(output_file, index=False)
        print(f"\nStatistics saved to: {output_file}")
    else:
        print("No files were processed.")


def generate_positive_samples_for_each_quarter():

    """
    Generate positive samples for each quarter by extracting all SE names from the SE_above_0.9 column
    and creating a new row for each value. The new row has the same report_id, DrugBankID, and hyperedge_label
    as the original row, but with a new SE name. The new SE name is stored in the SE_above_0.9 column.
    
    The output files are saved in the same directory as the input files, with the same filename but with
    "_positive_samples_SE_above_0.9_" appended to the filename.
    
    :return: None
    """
    
    # Define the directory containing the input files
    directory = 'dataset/condition123subsets'

    # Get all files in the directory
    files = os.listdir(directory)

    # Filter files that match the pattern (start with 'condition123' and end with 'step2.csv')
    pattern = r'^condition123.*step2\.csv$'
    target_files = [f for f in files if re.match(pattern, f)]

    # Process each matching file
    for input_file in target_files:
        # Build complete input file path
        input_path = os.path.join(directory, input_file)
        
        # Read CSV file
        df = pd.read_csv(input_path)
        
        # Create a list to store new rows
        new_rows = []
        
        # Iterate through each row in the original dataframe
        for _, row in df.iterrows():
            # Only process rows where SE_above_0.9 is not empty
            if row['SE_above_0.9'] != '[]':
                # Convert string lists to Python lists
                se_list = ast.literal_eval(row['SE_above_0.9'])
                drug_list = ast.literal_eval(row['DrugBankID_DrugRole'])
                
                # Extract all DrugBankIDs (second element only)
                drugbank_ids = [item[1] for item in drug_list]
                
                # Create a new row for each value in SE_above_0.9
                for se in se_list:
                    new_row = {
                        'report_id': row['report_id'],
                        'SE_above_0.9': se,
                        'DrugBankID': drugbank_ids,
                        'hyperedge_label': 1 
                    }
                    new_rows.append(new_row)
        
        # Create new dataframe
        new_df = pd.DataFrame(new_rows)
        
        # Build output filename
        # Extract year and quarter information from original filename
        match = re.search(r'_(20\d{2}Q[1-4])_', input_file)
        year_quarter = match.group(1) if match else ''
        
        new_df['time'] = year_quarter
        
        output_file = f'condition123_positive_samples_SE_above_0.9_{year_quarter}_step3.csv'
        output_path = os.path.join(directory, output_file)
        
        # Save results
        new_df.to_csv(output_path, index=False)
        print(f"Processing completed, results saved to: {output_path}")

    print("All files have been processed!")

def merge_and_deduplicate_csv(input_path, output_file):

    """
    Merge and deduplicate CSV files containing positive samples

    Parameters:
    input_path (str): Path to the directory containing the input CSV files
    output_file (str): Path to save the merged and deduplicated CSV file

    Returns:
    None

    This function will find all CSV files containing "positive_samples" in their names under the input directory,
    read them, merge them into one dataframe, shuffle the rows, drop duplicates based on SE_above_0.9 and DrugBankID,
    and save the result to a new CSV file.

    Before and after deduplication, the number of rows will be printed to the console.
    """
    # Find all CSV files containing "positive_samples"
    csv_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if "positive_samples" in file and file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    
    # Concatenate the dataframes
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle the rows
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Log the original number of rows
    print(f"Number of rows before deduplication: {len(merged_df)}")
    
    # Randomly deduplicate based on SE_above_0.9 and DrugBankID columns
    deduplicated_df = merged_df.drop_duplicates(subset=['SE_above_0.9', 'DrugBankID'], keep='first').reset_index(drop=True)
    
    # Log the number of rows after deduplication
    print(f"Number of rows after deduplication: {len(deduplicated_df)}")
    print(f"Number of deleted rows: {len(merged_df) - len(deduplicated_df)}")
    
    # Save the deduplicated dataframe
    deduplicated_df.to_csv(output_file, index=False)
 
 



def clean_DrugBankID():
    
    """
    Clean the DrugBankID column of the merged deduplicated condition123 dataset (2014Q3-2024Q3) by filtering out
    non-list values and strings that do not start with 'DB'. The resulting list is saved as a string representation.
    The output file has the same name as the input file.
    """
    
    file_path = 'dataset\\condition123subsets\\merged_deduplicated_condition123_2014Q3_2024Q3_step3.csv'
    df = pd.read_csv(file_path)
    
    def filter_db_ids(ids_str):
        if pd.isna(ids_str):
            return ids_str
        try:
            ids_list = ast.literal_eval(ids_str)
            if not isinstance(ids_list, list):
                raise ValueError
        except:
            return ids_str  # Return original string if unable to parse
        # Filter elements starting with 'DB'
        filtered_list = [id_ for id_ in ids_list if isinstance(id_, str) and id_.startswith('DB')]
        return filtered_list
    
    df['DrugBankID'] = df['DrugBankID'].apply(filter_db_ids)
    
    # Ensure that lists are saved as string representations
    df['DrugBankID'] = df['DrugBankID'].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    df.to_csv(file_path, index=False)


def delete_supersets_in_evaluation_subset():
    
    """
    Delete supersets in the merged deduplicated condition123 dataset (2014Q3-2024Q3) by grouping by SE_above_0.9 and sorting by the length of the DrugBankID_list column from small to large. 
    
    Then, for each group, check if the current set is a superset of any processed sets. If it is, delete the current row. 
    
    Finally, save the processed dataframe to a new CSV file and the deleted rows to another CSV file.
    """
    df = pd.read_csv('dataset\\condition123subsets\\merged_deduplicated_condition123_2014Q3_2024Q3_step3.csv')
    
    df['DrugBankID_list'] = df['DrugBankID'].apply(lambda x: ast.literal_eval(x))
    
    df['DrugBankID_list_length'] = df['DrugBankID_list'].apply(len)
    
    groups = df.groupby('SE_above_0.9')
    
    delete_indices = []
    
    for name, group in groups:
        group_sorted = group.sort_values(by='DrugBankID_list_length', ascending=True)
        
        processed = []
        
        for index, row in group_sorted.iterrows():
            current_set = set(row['DrugBankID_list'])
            
            if any(current_set.issuperset(p) for p in processed):
                delete_indices.append(index)
            else:
                processed.append(current_set)
    
    df_processed = df.drop(delete_indices)
    
    df_processed.to_csv(r'dataset\condition123subsets\positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv', index=False)
    
    df_deleted = df.loc[delete_indices]
    df_deleted.to_csv(r'dataset\condition123subsets\deleted_supersets_condition123_2014Q3_2024Q3_step4.csv', index=False)
    
    print(f"Deleted {len(delete_indices)} rows.")




def read_files_for_negative_sample_generation():
    
    """
    Reads in the required files for generating negative samples.

    Returns:
        tuple: (positive samples, SE data, DrugBank data)
    """
    print("Reading files...")
    # Read positive samples
    pos_samples = pd.read_csv(r'dataset\condition123subsets\positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv')
    
    # Read SE data
    se_data = pd.read_csv(r'dataset\dictionary\SE_similarity_2014Q3_2024Q3.csv')
    
    # Read DrugBank data
    drugbank_data = pd.read_csv(r'dataset\dictionary\Drugbank_ID_SMILE_all_structure links.csv')
    
    return pos_samples, se_data, drugbank_data

def generate_negative_samples(pos_samples, se_data, drugbank_data):
    
    """
    Generates negative samples from positive samples.

    Parameters:
    pos_samples (pandas.DataFrame): Positive samples DataFrame
    se_data (pandas.DataFrame): SE data DataFrame
    drugbank_data (pandas.DataFrame): DrugBank data DataFrame

    Returns:
    pandas.DataFrame: Negative samples DataFrame
    """
    
    negative_samples = []
    
    # Get all possible SE values
    print("Preparing SE values...")
    all_se_values = se_data['recommended_umls_cui_from_meddra'].unique()
    
    # Get all possible DrugBank IDs
    print("Preparing DrugBank IDs...")
    all_drugbank_ids = drugbank_data['DrugBank ID'].unique()
    
    # Process each positive sample row, add progress bar
    print("Generating negative samples...")
    for _, row in tqdm(pos_samples.iterrows(), total=len(pos_samples), desc="Generating negative samples"):
        # Create new negative sample row
        new_row = {}
        
        # Process report_id: add 'n' suffix
        new_row['report_id'] = f"{row['report_id']}n"
        
        # Process SE_above_0.9: randomly select a different SE value
        current_se = row['SE_above_0.9']
        available_se = all_se_values[all_se_values != current_se]
        new_row['SE_above_0.9'] = np.random.choice(available_se)
        
        # Process DrugBankID: randomly replace one drug ID
        current_drugs = ast.literal_eval(row['DrugBankID'])
        # Randomly select position to replace
        replace_position = np.random.randint(0, len(current_drugs))
        drug_to_replace = current_drugs[replace_position]
        
        # Exclude all drugs in current list from all possible drug IDs
        available_drugs = [drug for drug in all_drugbank_ids if drug not in current_drugs]
        
        # Randomly select new drug ID
        if available_drugs:
            new_drug = np.random.choice(available_drugs)
            new_drugs = current_drugs.copy()
            new_drugs[replace_position] = new_drug
            new_row['DrugBankID'] = str(new_drugs)
        
        # Set negative sample label
        new_row['hyperedge_label'] = -1
        
        negative_samples.append(new_row)
    
    # Create negative samples DataFrame
    print("Creating DataFrame...")
    neg_df = pd.DataFrame(negative_samples)
    return neg_df

def run_generate_negative_samples():
    # Read required files
    pos_samples, se_data, drugbank_data = read_files_for_negative_sample_generation()
    
    # Generate negative samples
    neg_samples = generate_negative_samples(pos_samples, se_data, drugbank_data)
    
    # Save results
    print("Saving results...")
    output_path = r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'
    neg_samples.to_csv(output_path, index=False)
    print(f"Negative samples saved to: {output_path}")





def deduplicate_negative_samples(input_file, output_file):
    
    """
    Remove duplicate rows from a CSV file and log the operation.

    Parameters:
    input_file (str): Path to the input CSV file from which duplicates will be removed.
    output_file (str): Path to the log file where the summary of the operation will be appended.

    This function reads the input CSV file, removes duplicate rows while keeping the first occurrence,
    and saves the deduplicated data back to the original file. It logs the number of original and final
    rows, as well as the number of duplicate rows removed, to the specified log file.
    """

    df = pd.read_csv(input_file)
    
    # Record the original row count
    original_count = len(df)
    
    # Delete duplicate rows while keeping the first occurrence
    df_deduplicated = df.drop_duplicates(keep='first')
    
    # Record the final row count
    final_count = len(df_deduplicated)
    
    # Save the deduplicated data back to the original file
    df_deduplicated.to_csv(input_file, index=False)
    
    # Log the operations
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Deduplicate Negative Samples - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Original Row Count: {original_count}\n")
        f.write(f"Delete Duplicate Rows: {final_count}\n")
        f.write(f"{original_count - final_count} rows were deleted\n")
        f.write("="*50 + "\n")

def check_if_negative_sample_happen_to_be_positive_sample():
    '''
    Check if negative samples happen to be positive samples
    '''

    # Read negative samples file
    negative_samples = pd.read_csv(r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv')

    # Read positive samples file
    positive_samples = pd.read_csv(r'dataset\condition123subsets\positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv')

    # Keep only the required columns
    negative_subset = negative_samples[['SE_above_0.9', 'DrugBankID']]
    positive_subset = positive_samples[['SE_above_0.9', 'DrugBankID']]

    # Create a list to store duplicate rows
    duplicates = []

    # Iterate through each row in negative samples with progress bar
    total_rows = len(negative_subset)
    for index, neg_row in tqdm(negative_subset.iterrows(), 
                            total=total_rows,
                            desc="Checking for duplicates",
                            ncols=100):
        # Check if this row appears in positive samples
        mask = (positive_subset['SE_above_0.9'] == neg_row['SE_above_0.9']) & \
            (positive_subset['DrugBankID'] == neg_row['DrugBankID'])
        
        if positive_subset[mask].shape[0] > 0:
            # If found a match, add this row to duplicates list
            duplicates.append(f"SE_above_0.9: {neg_row['SE_above_0.9']}, DrugBankID: {neg_row['DrugBankID']}")

    # Write results to file
    output_path = 'dataset/condition123subsets/statistics_negative_sample_happen_to_be_positive_sample.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("The following rows were found in negative samples but also appear in positive samples:\n\n")
        for duplicate in duplicates:
            f.write(duplicate + '\n')

    print(f"\nProcessing completed. Found {len(duplicates)} duplicates.")
    print(f"Results have been saved to: {output_path}")



def append_time_column():
    """
    Append time column to negative sample file.

    This function reads in the positive sample file and negative sample file, and appends the time column from the positive sample file to the negative sample file. If the negative sample file has more rows than the positive sample file, it will truncate the negative sample file to the same number of rows.

    Parameters:
    None

    Returns:
    None
    """
    positive_file = 'dataset/condition123subsets/positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'
    negative_file = 'dataset/condition123subsets/negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'
    
    # Positive sample dtype
    positive_dtype = {
        'ID': int,
        'encoding1': str,
        'encoding2': str,
        'value': float,
        'time': str
    }
    
    # Read positive samples and get the number of rows
    positive_df = pd.read_csv(positive_file, dtype=positive_dtype)
    n_rows = len(positive_df)
    
    # Assiging negative sample dtype
    negative_dtype = {
        'ID': int,
        'encoding1': str,
        'encoding2': str,
        'value': float,
        'time': str
    }
    
    # Read negative samples
    negative_df = pd.read_csv(negative_file, dtype=negative_dtype)
    
    # If the number of rows in the negative sample file is greater than n_rows, truncate it to n_rows
    if len(negative_df) > n_rows:
        negative_df = negative_df.iloc[:n_rows]
    
    # Copy the 'time' column from the positive sample file
    negative_df['time'] = positive_df['time']
    
    # Overwrite the negative sample file
    negative_df.to_csv(negative_file, index=False)


def split_csv_by_time(input_csv_path):
    """
    Split a CSV file into multiple CSV files based on the time column.

    Parameters:
        input_csv_path (str): Path to the input CSV file.

    Returns:
        None

    Notes:
        This function assumes that the input CSV file has a time column.
        The output files will be saved in a folder named `evaluation_subset` under the current working directory.
    """
    filename = os.path.basename(input_csv_path)
    
    # Determine sample type
    if 'positive_samples' in filename:
        sample_type = 'positive'
    elif 'negative_samples' in filename:
        sample_type = 'negative'
    else:
        sample_type = 'unknown'
    
    # Extract condition from filename
    parts = filename.split('_')
    condition_part = next((part for part in parts if part.startswith('condition')), None)
    if condition_part:
        condition = condition_part
    else:
        condition = 'condition123'
    
    se_above = 'SE_above_0.9'  
    
    df = pd.read_csv(input_csv_path)
    
    
    if 'time' not in df.columns:
        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if time_col:
            df.rename(columns={time_col: 'time'}, inplace=True)
        else:
            print("Error: CSV file does not contain 'time' column.")
            return
    
    # Get unique values of the 'time' column
    time_values = df['time'].unique()
    
    target_folder = os.path.join(os.getcwd(), 'dataset', 'evaluation_subset')
    os.makedirs(target_folder, exist_ok=True)
    
    # Get unique values of the 'time' column
    for time_val in time_values:
        df_time = df[df['time'] == time_val]
        
        new_filename = f"{time_val}_{sample_type}_samples_{condition}_{se_above}.csv"
        
        output_path = os.path.join(target_folder, new_filename)
        df_time.to_csv(output_path, index=False)
    
    print(f"Saved to {target_folder} ")

def modify_csv_files(directory):

    """
    Modify CSV files in the given directory by deleting specific columns.

    Parameters:
        directory (str): The directory containing the CSV files to modify.

    Returns:
        None

    Notes:
        This function will delete the 'DrugBankID_list' and 'DrugBankID_list_length' columns from files with names containing 'positive_samples',
        and delete the 'time_positive' column from files with names containing 'negative_samples'.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            
            if 'positive_samples' in filename:
                if 'DrugBankID_list' in df.columns:
                    df.drop(columns=['DrugBankID_list', 'DrugBankID_list_length'], inplace=True)
                df.to_csv(file_path, index=False)
                print(f"Processed and saved: {filename}")
            
            elif 'negative_samples' in filename:
                if 'time_positive' in df.columns:
                    df.drop(columns=['time_positive'], inplace=True)
                df.to_csv(file_path, index=False)
                print(f"Processed and saved: {filename}")


def generate_subset_for_merged_positive_samples(min_drug_length, max_drug_length, min_se_count, max_se_count):
    
    """
    Generate a subset of positive samples with drug length between min_drug_length and max_drug_length, and SE count between min_se_count and max_se_count.

    Parameters:
        min_drug_length (int): The minimum drug length.
        max_drug_length (int): The maximum drug length.
        min_se_count (int): The minimum SE count.
        max_se_count (int): The maximum SE count.

    Returns:
        None

    Notes:
        This function will read the input file, filter out rows based on drug length and SE count, and save the results to a new CSV file.
    """
    try:
        file_path = 'dataset/condition123subsets/positive_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'
        print(f"Reading file from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: Input file not found at {file_path}")
            return
            
        df = pd.read_csv(file_path)
        print(f"Read {len(df)} rows from input file")
        
        df['row_index'] = range(1, len(df) + 1)
        filtered_df = df[(df['DrugBankID_list_length'] >= min_drug_length) & 
                        (df['DrugBankID_list_length'] <= max_drug_length)]
        print(f"After drug length filter: {len(filtered_df)} rows")
        
        se_counts = filtered_df['SE_above_0.9'].value_counts()
        valid_se = se_counts[(se_counts >= min_se_count) & 
                            (se_counts <= max_se_count)].index
        print(f"Found {len(valid_se)} valid SE entries")
        
        final_df = filtered_df[filtered_df['SE_above_0.9'].isin(valid_se)]
        print(f"Final dataset size: {len(final_df)} rows")
        
        output_dir = f'dataset/evaluation_subset/subset_drug{min_drug_length}-{max_drug_length}_SE{min_se_count}-{max_se_count}'
        os.makedirs(output_dir, exist_ok=True)
        
        output_file_path = os.path.join(output_dir, 'positive_samples_2014Q3_2024Q3_step6.csv')
        final_df.to_csv(output_file_path, index=False)
        print(f"Results saved to: {output_file_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def fetch_negative_samples(positive_csv_path, negative_csv_path= r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'):
    
    """
    Fetch negative samples from a given CSV file based on the row indices in another CSV file

    Parameters:
        positive_csv_path (str): Path to the CSV file containing the positive samples
        negative_csv_path (str, optional): Path to the CSV file containing the negative samples. Defaults to r'dataset\condition123subsets\negative_samples_merged_deduplicated_superset_removed_condition123_2014Q3_2024Q3_step5.csv'

    Returns:
        None

    Notes:
        This function reads the input CSV file, extracts the row indices from the 'row_index' column, and finds the matching rows in the negative samples CSV file. The matched rows are then saved to a new CSV file with the same path as the input file but with 'negative' instead of 'positive' in the filename.
    """
    positive_df = pd.read_csv(positive_csv_path)
    
    # Extract row indices from the 'row_index' column
    list_A = positive_df['row_index'].tolist()
    
    # Read the negative samples CSV file
    negative_df = pd.read_csv(negative_csv_path)
    
    # Find rows with matching row indices
    matched_rows = negative_df[negative_df['row_index'].isin(list_A)]
    
    # New CSV file path
    positive_dir = os.path.dirname(positive_csv_path)
    positive_filename = os.path.basename(positive_csv_path)
    negative_filename = positive_filename.replace('positive', 'negative')
    negative_output_path = os.path.join(positive_dir, negative_filename)
    
    # Save the matched rows to a new CSV file
    matched_rows.to_csv(negative_output_path, index=False)
    
    print(f"Fetched negative samples saved to: {negative_output_path}")
    





def process_dataset_folder(folder_path):
    """
    Process CSV files in the specified folder path.
    
    Parameters:
        folder_path (str): Path to the folder containing CSV files
    """
    # Check for positive and negative sample files
    pos_file = os.path.join(folder_path, 'positive_samples_2014Q3_2024Q3_step6.csv')
    neg_file = os.path.join(folder_path, 'negative_samples_2014Q3_2024Q3_step6.csv')
    
    # Process each file if it exists
    for file_path in [pos_file, neg_file]:
        if os.path.exists(file_path):
            split_csv_by_time2(file_path, folder_path)
    
    # Create merged_subset folder and move files
    merged_folder = os.path.join(folder_path, 'merged_subset')
    os.makedirs(merged_folder, exist_ok=True)
    
    # Move files containing '_step6'
    for filename in os.listdir(folder_path):
        if '_step6' in filename and filename.endswith('.csv'):
            src_path = os.path.join(folder_path, filename)
            dst_path = os.path.join(merged_folder, filename)
            shutil.move(src_path, dst_path)

def split_csv_by_time2(input_csv_path, target_folder):
    """
    Split a CSV file into multiple CSV files based on the time column.
    
    Parameters:
        input_csv_path (str): Path to the input CSV file
        target_folder (str): Path where output files will be saved
    """
    filename = os.path.basename(input_csv_path)
    
    # Determine sample type
    if 'positive_samples' in filename:
        sample_type = 'positive'
    elif 'negative_samples' in filename:
        sample_type = 'negative'
    else:
        sample_type = 'unknown'
    
    # Extract condition from filename
    parts = filename.split('_')
    condition_part = next((part for part in parts if part.startswith('condition')), None)
    condition = condition_part if condition_part else 'condition123'
    
    se_above = 'SE_above_0.9'
    
    df = pd.read_csv(input_csv_path)
    
    if 'time' not in df.columns:
        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        if time_col:
            df.rename(columns={time_col: 'time'}, inplace=True)
        else:
            print(f"Error: CSV file {filename} does not contain 'time' column.")
            return
    
    # Process each unique time value
    for time_val in df['time'].unique():
        df_time = df[df['time'] == time_val]
        new_filename = f"{time_val}_{sample_type}_samples_{condition}_{se_above}.csv"
        output_path = os.path.join(target_folder, new_filename)
        df_time.to_csv(output_path, index=False)
    
    print(f"Split files saved to {target_folder}")



def hypergraph_to_graph(file_path):

    """
    Convert hypergraph data to graph data by creating edges between all pairs of nodes in each hyperedge.

    Parameters:
    file_path (str): Path to the input CSV file containing hypergraph data

    Returns:
    pandas.DataFrame: DataFrame containing the converted graph data with columns:
        - source: The source node ID
        - target: The target node ID
        - SE_label: The SE label of the edge
        - label: The label of the edge
        - report_id: The report ID of the edge
    """
    
    # Read HGNN dataset
    df = pd.read_csv(file_path)
    
    # Store converted edges
    edges = []
    edge_labels = []
    
    for _, row in df.iterrows():
        # Convert string-formatted drug IDs to actual lists
        drug_ids = eval(row['DrugBankID'])
        hyperedge_label = row['hyperedge_label']
        se_label = row['SE_above_0.9']
        report_id = row['report_id']
        
        # Create edges for each pair of nodes
        for drug1, drug2 in combinations(drug_ids, 2):
            edges.append((drug1, drug2))
            # Save edge labels
            edge_labels.append({
                'SE_label': se_label,
                'label': hyperedge_label,
                'report_id': report_id
            })
    
    # Create a new DataFrame to store the converted graph data
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_df['SE_label'] = [label['SE_label'] for label in edge_labels]
    edge_df['label'] = [label['label'] for label in edge_labels]
    edge_df['report_id'] = [label['report_id'] for label in edge_labels]
    
    return edge_df

def hgnn2gnn_clique_expansion_for_one_csv_file(file_path, output_dir):

    """
    Convert a single HGNN dataset CSV file to a graph dataset using clique expansion.

    Parameters:
        file_path (str): Path to the input CSV file containing HGNN dataset
        output_dir (str): Path to the output directory to save the converted graph data

    Returns:
        None

    Notes:
        The converted graph data will be saved as a CSV file with the same filename as the input
        file but with "_clique_expansion" appended to the filename.
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file name and generate output file name
    file_name = os.path.basename(file_path)
    output_file_name = file_name.replace('.csv', '_clique_expansion.csv')
    output_path = os.path.join(output_dir, output_file_name)
    
    # Convert hypergraph data to graph data 
    edge_df = hypergraph_to_graph(file_path)
    edge_df.to_csv(output_path, index=False)
    
    print(f"Converted GNN dataset saved to: {output_path}")
    print(f"- Number of edges: {len(edge_df)}")
    print(f"- Unique drugs: {len(set(edge_df['source'].unique()) | set(edge_df['target'].unique()))}")

def hgnn2gnn_clique_expansion_in_a_given_directory(input_dir, output_dir):

    """
    Convert all HGNN dataset CSV files in a given directory to graph datasets using clique expansion.

    Parameters:
        input_dir (str): Path to the input directory containing HGNN dataset CSV files
        output_dir (str): Path to the output directory to save the converted graph data

    Returns:
        None

    Notes:
        The converted graph data will be saved as a CSV file with the same filename as the input
        file but with "_clique_expansion" appended to the filename.
    """
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # Ignore subdirectories
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            print(f"Processing file: {file_name}")
            hgnn2gnn_clique_expansion_for_one_csv_file(file_path, output_dir)
            print()




def parse_drugbank_xml(xml_path):
    
    """
    Parse a DrugBank XML file and extract the DrugBank ID and SMILES
    information for each drug.

    Parameters:
        xml_path (str): Path to the DrugBank XML file

    Returns:
        pd.DataFrame: A DataFrame containing the extracted information

    Notes:
        The returned DataFrame will have two columns: 'drugbank_id' and 'smiles'.
        The 'smiles' column will contain NaN values for drugs that do not have
        SMILES information in the XML file.
    """
    # Register namespaces
    ns = {'db': 'http://www.drugbank.ca'}
    
    # Parse the XML
    print("Starting to parse XML file...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create a list to store the results
    drug_data = []
    
    # Iterate over all 'drug' elements
    for drug in root.findall('db:drug', ns):
        drugbank_id = drug.find('db:drugbank-id', ns).text
        
        # Get SMILES
        calc_props = drug.find('db:calculated-properties', ns)
        smiles = None
        
        if calc_props is not None:
            for prop in calc_props.findall('db:property', ns):
                kind = prop.find('db:kind', ns).text
                if kind == 'SMILES':
                    smiles = prop.find('db:value', ns).text
                    break
        
        # Store all DrugBankIDs
        drug_data.append({
            'drugbank_id': drugbank_id,
            'smiles': smiles
        })
        
    df = pd.DataFrame(drug_data)
    
    print(f"\n{len(df)} drugs were found")
    print(f"DrugBankID with SMILES: {df['smiles'].notna().sum()}")
    
    return df

def generate_BankDrugID2SMILES():
    
    
    """
    Parse the DrugBank XML file and generate a CSV mapping from DrugBank IDs to SMILES strings.

    The CSV file is saved to 'dataset/dictionary/DrugBankID2SMILES.csv'.

    :return: None
    """
    xml_path = r'dataset\dictionary\DrugBankFullDataBase.xml'
    
    try:
        df = parse_drugbank_xml(xml_path)
        
        print("\nHead of the DataFrame:")
        print(df.head())
        
        output_file = r'dataset\dictionary\DrugBankID2SMILES.csv'
        df.to_csv(output_file, index=False)
        print(f"\nFile saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

    


