# Import libraries
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def visualize_training_process_for_clique_expansion(base_path):
    
    """
    Visualize the training process for clique expansion model.

    This function iterates through all CSV files in the given base path, reads the CSV file, and
    generates a plot showing the training metrics over epochs. The plot is then saved to a PNG file
    with the same filename as the CSV file but with a .png extension.

    Parameters:
        base_path (str): The path to the directory containing the CSV files.

    Returns:
        None
    """
    for filename in os.listdir(base_path):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(base_path, filename)
            
            output_image_path = os.path.join(base_path, filename.replace(".csv", ".png"))
            
            df = pd.read_csv(input_csv_path)

            last_row = df.iloc[-1]

            epochs = df['Epoch']

            plt.figure(figsize=(12, 7))

            plt.plot(epochs, df['Loss'], label='Loss')
            plt.plot(epochs, df['Val_AUC_DD'], label='Val_AUC_DD')
            plt.plot(epochs, df['Val_AUC_DSE'], label='Val_AUC_DSE')
            plt.plot(epochs, df['Precision'], label='Precision')
            plt.plot(epochs, df['F1_Score'], label='F1_Score')
            plt.plot(epochs, df['AUC'], label='AUC')
            plt.plot(epochs, df['AUPRC'], label='AUPRC')

            plt.title('Metrics over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Value')

            plt.legend()

            final_results_text = "Final Results:\n"
            for col, value in last_row.items():
                if col != 'Epoch':  
                    final_results_text += f"{col}: {value:.4f}\n"

            plt.text(0.5, 0.05, final_results_text, 
                    fontsize=10, ha='right', va='bottom', color='black', 
                    transform=plt.gca().transAxes,  
                    bbox=dict(facecolor='white', alpha=0.8))  

            plt.savefig(output_image_path, dpi=300, transparent=True, bbox_inches='tight')
            
            plt.close()
            
            print(f"Processed {filename} and saved image to {output_image_path}")




def visualize_training_process_for_MLP(base_path):
    
    """
    Visualize the training process for MLP model.

    This function iterates through all CSV files in the given base path, reads the CSV file, and
    generates a plot showing the training metrics over epochs. The plot is then saved to a PNG file
    with the same filename as the CSV file but with a .png extension.

    Parameters:
        base_path (str): The path to the directory containing the CSV files.

    Returns:
        None
    """
    
    for filename in os.listdir(base_path):
        if filename.endswith(".csv"):
            input_csv_path = os.path.join(base_path, filename)
            
            output_image_path = os.path.join(base_path, filename.replace(".csv", ".png"))
            
            df = pd.read_csv(input_csv_path)

            last_row = df.iloc[-1]

            epochs = df['Epoch']

            plt.figure(figsize=(12, 7))

            plt.plot(epochs, df['Train_Loss'], label='Loss')
            plt.plot(epochs, df['Val_Precision'], label='Val_Precision')
            plt.plot(epochs, df['Val_F1'], label='Val_F1')
            plt.plot(epochs, df['Val_AUC'], label='Val_AUC')
            plt.plot(epochs, df['Val_PRAUC'], label='Val_AUPRC')



            plt.title('Metrics over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Value')

            plt.legend()

            final_results_text = "Final Results:\n"
            for col, value in last_row.items():
                if col != 'Epoch':  
                    final_results_text += f"{col}: {value:.4f}\n"

            plt.text(0.5, 0.05, final_results_text, 
                    fontsize=10, ha='right', va='bottom', color='black', 
                    transform=plt.gca().transAxes,  
                    bbox=dict(facecolor='white', alpha=0.8))  

            plt.savefig(output_image_path, dpi=300, transparent=True, bbox_inches='tight')
            
            plt.close()
            
            print(f"Processed {filename} and saved image to {output_image_path}")

def merge_multiple_train_csvs(input_dir):

    """
    Merge multiple training CSV files containing metrics with different seeds into a single CSV file.

    Parameters:
        input_dir (str): The directory path containing the training CSV files.

    Returns:
        None

    Notes:
        The CSV filenames are expected to contain the seed number in the format 'seed{number}.csv'.
        The output CSV file will be named 'combined_results.csv' and will be saved in the input directory.
    """

    path = input_dir

    result_df = pd.DataFrame()

    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            match = re.search(r'seed(\d+)\.csv', filename)
            if match:
                seed = int(match.group(1))
                
                filepath = os.path.join(path, filename)
                df = pd.read_csv(filepath)
                
                last_row = df.iloc[-1].to_frame().T
                
                last_row['Seed'] = seed
                
                result_df = pd.concat([result_df, last_row], ignore_index=True)

    result_df = result_df.round(4)

    mean_values = result_df.iloc[:, :-1].mean().round(4)  
    std_values = result_df.iloc[:, :-1].std().round(4)    

    mean_values_df = mean_values.to_frame().T
    mean_values_df['Seed'] = 'Mean'  

    std_values_df = std_values.to_frame().T
    std_values_df['Seed'] = 'Std_dev'  

    result_df = pd.concat([result_df, mean_values_df], ignore_index=True)
    result_df = pd.concat([result_df, std_values_df], ignore_index=True)

    output_path = os.path.join(path, 'combined_results.csv')
    result_df.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")
    print("Updated File Content:")
    print(result_df)