import pandas as pd
import os
import glob

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