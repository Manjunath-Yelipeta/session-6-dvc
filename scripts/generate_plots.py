import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_latest_metrics_file(logs_dir):
    pattern = os.path.join(logs_dir, "train/runs/*/csv/version_0/metrics.csv")
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No metrics.csv files found in {logs_dir}")
    return max(files, key=os.path.getctime)

def plot_accuracy(df, output_dir):
    plt.figure(figsize=(10, 5))
    # Only plot metrics that exist in the data
    if 'train/acc_step' in df.columns:
        plt.plot(df['step'], df['train/acc_step'], label='Train Accuracy (Step)')
    if 'val/acc_step' in df.columns:
        plt.plot(df['step'], df['val/acc_step'], label='Validation Accuracy (Step)')
    
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy (Step)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_step_plot.png'))
    plt.close()

def plot_loss(df, output_dir):
    plt.figure(figsize=(10, 5))
    # Only plot metrics that exist in the data
    if 'train/loss_step' in df.columns:
        plt.plot(df['step'], df['train/loss_step'], label='Train Loss (Step)')
    if 'val/loss_step' in df.columns:
        plt.plot(df['step'], df['val/loss_step'], label='Validation Loss (Step)')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss (Step)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_step_plot.png'))
    plt.close()

def plot_confusion_matrix(df, output_dir, prefix):
    max_epoch = df['epoch'].max()
    cm_columns = [f"{prefix}_confusion_matrix_{i}" for i in range(4)]
    if all(col in df.columns for col in cm_columns):
        cm_data = df[df['epoch'] == max_epoch][cm_columns].iloc[0].values
        cm = np.array(cm_data).reshape((2, 2)).astype(int)  # Convert to integers
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{prefix.capitalize()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'{prefix}_confusion_matrix.png'))
        plt.close()
    else:
        print(f"{prefix.capitalize()} confusion matrix data not found in the CSV file")

if __name__ == "__main__":
    # Assuming the script is in the 'scripts' directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    output_dir = os.path.join(project_root, "plots")
    os.makedirs(output_dir, exist_ok=True)

    latest_metrics_file = find_latest_metrics_file(logs_dir)
    print(f"Latest metrics file: {latest_metrics_file}")

    # Read the CSV file
    df = pd.read_csv(latest_metrics_file)
    print("Columns in the DataFrame:", df.columns)

    # Plot only step-based metrics
    plot_accuracy(df, output_dir)
    plot_loss(df, output_dir)

    print(f"Plots saved in {output_dir}")
