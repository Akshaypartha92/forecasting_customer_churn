import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def loader_processor_single_f(filepath):
    """
    Loads and cleans a single monthly churn CSV.
    Adds a 'Month' column and standardizes the churn column name.
    """
    df = pd.read_csv(filepath)
    month = int(filepath.split("_")[-1].split(".")[0])
    df["Month"] = month

    # Standardize churn column
    churn_col = [col for col in df.columns if col.lower().startswith("churned_")]
    if churn_col:
        df = df.rename(columns={churn_col[0]: "is_churned"})
    
    return df


def load_raw_data(raw_folder_path):
    """
    Loads all synthetic churn data from raw folder and returns combined DataFrame.
    """
    file_paths = glob.glob(os.path.join(raw_folder_path, "synthetic_churn_data_month_*.csv"))
    file_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    dataframes = list(map(loader_processor_single_f, file_paths))
    return pd.concat(dataframes, ignore_index=True)

def plot_feature_comparison(df, x_col, y_col, palette=['#F08080', '#20B2AA']):
    """
    Plots a comparison bar chart showing the average of y_col grouped by x_col.
    
    """
    group_avg = df.groupby(x_col)[y_col].mean().reset_index()

    plt.figure(figsize=(7, 5))
    sns.barplot(data=group_avg, x=x_col, y=y_col, palette=palette)
    
    # Add value labels
    for i, val in enumerate(group_avg[y_col]):
        label = f"${val:.2f}" if 'amount' in y_col.lower() or 'charge' in y_col.lower() else f"{val:.0f} months"
        plt.text(i, val + (val * 0.02), label, ha='center')
    
    # Titles and labels
    plt.title(f"Average {y_col} by {x_col}")
    plt.ylabel(y_col)
    plt.xlabel(x_col)
    plt.tight_layout()
    plt.show()