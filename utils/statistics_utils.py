# Import libraries
import os
import ast
import logging
import pandas as pd
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up matplotlib style
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300
})



def plot_se_statistics(df_path, output_dir):
    """
    Plot statistics for side effects in a given CSV file.

    Parameters:
        df_path (str): Path to the input CSV file.
        output_dir (str): Path to the output directory.

    Returns:
        None

    Notes:
        This function assumes that the input CSV file has a column named 'SE_above_0.9'.
        The output files will be saved in a folder named `evaluation_subset` under the current working directory.
    """
    
    try:
        logging.info(f"Processing SE statistics for {df_path}")
        
        if not os.path.exists(df_path):
            logging.error(f"File not found: {df_path}")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(df_path)
        
        if 'SE_above_0.9' not in df.columns:
            logging.error(f"Column 'SE_above_0.9' not found in {df_path}")
            return
        
        se_counts = df['SE_above_0.9'].value_counts()
        se_stats = se_counts.value_counts().reset_index(name='num_SE')
        se_stats.columns = ['num_occurence', 'num_SE']
        se_stats = se_stats.sort_values('num_occurence')
        
        prefix = 'positive' if 'positive' in df_path else 'negative'
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(se_stats['num_occurence'], se_stats['num_SE'], 
               color='#2878B5', alpha=0.8)
        
        ax.set_xlabel('Number of Occurrences')
        ax.set_ylabel('Number of Side Effects')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{prefix}_samples_SE_stratified.png')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        stats_path = os.path.join(output_dir, f'{prefix}_samples_SE_statistics.csv')
        se_stats.to_csv(stats_path, index=False)
        logging.info(f"Successfully saved SE statistics for {prefix} samples")
        
    except Exception as e:
        logging.error(f"Error processing {df_path}: {str(e)}")

def plot_drug_statistics(df_path, output_dir):
    """
    Plot statistics for number of drugs per record in a given CSV file.

    Parameters:
        df_path (str): Path to the input CSV file.
        output_dir (str): Path to the output directory.

    Returns:
        None

    Notes:
        This function assumes that the input CSV file has a column named 'DrugBankID'.
        The output files will be saved in a folder named `evaluation_subset` under the current working directory.
    """
    try:
        logging.info(f"Processing drug statistics for {df_path}")
        
        if not os.path.exists(df_path):
            logging.error(f"File not found: {df_path}")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(df_path)
        
        if 'DrugBankID' not in df.columns:
            logging.error(f"Column 'DrugBankID' not found in {df_path}")
            return
        
        def count_drugs(drug_str):
            try:
                drug_list = ast.literal_eval(drug_str)
                return len([x for x in drug_list if x != 'none'])
            except:
                return 0
        
        drug_counts = df['DrugBankID'].apply(count_drugs)
        drug_stats = drug_counts.value_counts().reset_index(name='num_records')
        drug_stats.columns = ['num_drug', 'num_records']
        drug_stats = drug_stats.sort_values('num_drug')
        
        prefix = 'positive' if 'positive' in df_path else 'negative'
        
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(drug_stats['num_drug'], drug_stats['num_records'],
               color='#2878B5', alpha=0.8)
        
        ax.set_xlabel('Number of Drugs per Record')
        ax.set_ylabel('Number of Records')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{prefix}_samples_drug_stratified.png')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        stats_path = os.path.join(output_dir, f'{prefix}_samples_drug_statistics.csv')
        drug_stats.to_csv(stats_path, index=False)
        logging.info(f"Successfully saved drug statistics for {prefix} samples")
        
    except Exception as e:
        logging.error(f"Error processing {df_path}: {str(e)}")
        
        



def se_positive_samples_stratified_statistics(input_csv_path, output_csv_path):
    """
    Compute stratified statistics for positive samples based on the number of occurrences of each side effect.

    Parameters:
        input_csv_path (str): Path to the input CSV file containing the side effect statistics.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        None

    Notes:
        The output CSV file will contain two columns: 'num_occurences' and 'num_SE'. The 'num_occurences' column contains the stratified intervals,
        and the 'num_SE' column contains the total number of side effects in each interval.
    """
    
    
    df = pd.read_csv(input_csv_path)
    
    bins = [0, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 5000]
    labels = ['1-10', '11-20', '21-30', '31-40', '41-50', '51-100', '101-200', '201-500', '501-1000', '1001-5000']
    
    df['num_occurences'] = pd.cut(df['num_occurence'], bins=bins, labels=labels, right=False)
    
    stratified_sum = df.groupby('num_occurences')['num_SE'].sum().reset_index()
    
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stratified_sum.to_csv(output_csv_path, index=False)



def se_negative_samples_stratified_statistics(input_csv_path, output_csv_path):
    """
    Compute stratified statistics for negative samples based on the number of occurrences of each side effect.

    Parameters:
        input_csv_path (str): Path to the input CSV file containing the side effect statistics.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        None

    Notes:
        The output CSV file will contain two columns: 'num_occurences' and 'num_SE'. The 'num_occurences' column contains the stratified intervals,
        and the 'num_SE' column contains the total number of side effects in each interval.
    """

    df = pd.read_csv(input_csv_path)
    
    bins = [0, 10, 20, 30, 40, 50]
    labels = ['1-10', '11-20', '21-30', '31-40', '41-50']
    
    df['num_occurences'] = pd.cut(df['num_occurence'], bins=bins, labels=labels, right=False)
    
    stratified_sum = df.groupby('num_occurences')['num_SE'].sum().reset_index()
    
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stratified_sum.to_csv(output_csv_path, index=False)
    
    


def drug_positive_and_negative_samples_stratified_statistics(input_csv_path, output_csv_path):
    """
    Compute stratified statistics for the number of drugs per record in a dataset of positive and negative samples.

    Parameters:
        input_csv_path (str): Path to the input CSV file containing drug statistics.
        output_csv_path (str): Path to save the output CSV file with stratified statistics.

    Returns:
        None

    Notes:
        The function reads a CSV file, stratifies the records based on the number of drugs per record into defined intervals,
        and outputs a CSV file with the stratified sum of records for each interval.
    """

    df = pd.read_csv(input_csv_path)
    
    bins = [1, 2, 5, 10, 15, 20, 30, 40, 50, 100, float('inf')]
    labels = ['1', '2-5', '6-10', '11-15', '16-20', '21-30', '31-40', '41-50', '51-100', '101+']
    
    df['num_drugs_per_record'] = pd.cut(df['num_drug'], bins=bins, labels=labels, right=False)
    
    stratified_sum = df.groupby('num_drugs_per_record')['num_records'].sum().reset_index()
    
    output_dir = os.path.dirname(output_csv_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stratified_sum.to_csv(output_csv_path, index=False)






def zoom_in_se_positive_samples():
    """
    Reads two CSV files and plots two different bar charts based on the data.
    The first plot is a bar chart for num_occurence in the range 1-20.
    The second plot is a bar chart for num_occurences excluding the '1-10' range.
    The generated images are saved to the specified output directory.
    """
    
    try:
        # Paths to the CSV files
        positive_samples_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\positive_samples_SE_statistics.csv'
        stratified_samples_path = r'dataset\condition123subsets\statistics\2014Q3_2024Q3\stratified_statistics\SE_positive_samples.csv'
        
        # Output directory for saving images
        output_dir = r'dataset\condition123subsets\statistics\2014Q3_2024Q3'
        
        # Check if files exist
        if not os.path.exists(positive_samples_path) or not os.path.exists(stratified_samples_path):
            logging.error("One or both CSV files not found.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the first CSV file
        df1 = pd.read_csv(positive_samples_path)
        
        # Filter data for num_occurence between 1 and 20
        df1_filtered = df1[(df1['num_occurence'] >= 1) & (df1['num_occurence'] <= 20)]
        
        # Plot the first bar chart
        plt.figure(figsize=(4, 3))
        plt.bar(df1_filtered['num_occurence'], df1_filtered['num_SE'], color='blue', alpha=0.7)
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Number of Side Effects')
        # plt.title('Number of Side Effects for 1-20 Occurrences')
        plt.xticks(range(0, 21, 2))
        # plt.grid(True)
        
        # Save the first plot
        plot1_path = os.path.join(output_dir, 'positive_samples_se_1-20.png')
        plt.savefig(plot1_path, bbox_inches='tight')
        plt.close()
        
        # Read the second CSV file
        df2 = pd.read_csv(stratified_samples_path)
        
        # Exclude the '1-10' range
        df2_filtered = df2[df2['num_occurences'] != '1-10']
        
        # Plot the second bar chart
        plt.figure(figsize=(4, 3))
        plt.bar(df2_filtered['num_occurences'], df2_filtered['num_SE'], color='green', alpha=0.7)
        plt.xlabel('Number of Occurrences')
        plt.ylabel('Number of Side Effects')
        # plt.title('Number of Side Effects for Different Occurrence Ranges (Excluding 1-10)')
        plt.xticks(rotation=45)
        # plt.grid(True)
        
        # Save the second plot
        plot2_path = os.path.join(output_dir, 'positive_samples_se_excluding_1-10.png')
        plt.savefig(plot2_path, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Plots saved successfully to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")




def modify_and_save_csv1(file_path):
    """
    Modify and save a CSV file.

    Parameters:
        file_path (str): Path to the input CSV file.

    Returns:
        None

    Notes:
        This function assumes that the input CSV file has a column named 'file_name'.
        The output CSV file will be saved in the same directory as the input file, with the same file name.
    """
    df = pd.read_csv(file_path)

    df.rename(columns={'file_name': 'time'}, inplace=True)

    df['time'] = df['time'].str.replace('condition123_subset_', '').str.replace('_step2.csv', '')

    data_without_sum = df.iloc[:-1, 1:]  

    mean_values = data_without_sum.mean()
    std_values = data_without_sum.std()

    mean_row = pd.DataFrame([['Mean'] + mean_values.tolist()], columns=df.columns)
    std_row = pd.DataFrame([['Std_dev'] + std_values.tolist()], columns=df.columns)

    df = pd.concat([df, mean_row, std_row], ignore_index=True)

    df.to_csv(file_path, index=False)



def modify_and_save_csv2(file_path):
    '''
    Clean up condition123_stratified_per_quarter.csv
    '''
    df = pd.read_csv(file_path)

    df.rename(columns={'file_name': 'time'}, inplace=True)

    df['time'] = df['time'].str.replace('condition123_subset_', '').str.replace('.csv', '')


    data_without_total = df.iloc[:-1, 1:] 

    mean_values = data_without_total.mean()
    std_values = data_without_total.std()

    mean_row = pd.DataFrame([['Mean'] + mean_values.tolist()], columns=df.columns)
    std_row = pd.DataFrame([['Std_dev'] + std_values.tolist()], columns=df.columns)

    df = pd.concat([df, mean_row, std_row], ignore_index=True)

    df.to_csv(file_path, index=False)




