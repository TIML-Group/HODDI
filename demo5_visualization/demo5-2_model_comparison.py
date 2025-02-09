# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_metrics(csv_file):
    
    """
    Plot model metrics from a CSV file.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing the model metrics.


    Returns
    -------
    None
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    
    # Get unique models
    models = df[df['Statistics'] == 'Mean']['Model'].unique()
    metrics = ['Precision','F1', 'AUC', 'PRAUC']
    
    # Plot setup
    n_metrics = len(metrics)
    n_models = len(models)
    width = 0.25
    x = np.arange(n_metrics)
    colors = ['#4e79a7', '#f28e2b', '#59a22f', '#af7aa1', '#edc948']  
    
    plt.figure(figsize=(10, 6))
    
    # Set font and font size
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 22
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        means = model_data[model_data['Statistics'] == 'Mean'].iloc[0]
        stds = model_data[model_data['Statistics'] == 'Std_dev'].iloc[0]
        
        heights = [means[metric] * 100 for metric in metrics]
        yerr = [stds[metric] * 100 for metric in metrics]
        
        plt.bar(x + (i - n_models/2 + 0.5) * width, heights, width, 
               label=model, color=colors[i], yerr=yerr, capsize=5)
    
    plt.ylabel('Percentage', fontsize=24)
    plt.xticks(x, metrics, fontsize=24)
    plt.legend(loc='best', fontsize=22)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.ylim(70, 100)
    
    plt.tight_layout()
    # plt.savefig('model/merged_result/model_comparison_val.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.savefig('model/merged_result/model_comparison_test.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

if __name__ == '__main__':
    # plot_model_metrics('model/merged_result/merged_val_data.csv')
    plot_model_metrics('model/merged_result/merged_test_data.csv')