import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

def apply_smote(X, y, scale=False):
    """
    Applies SMOTE to balance the target classes, specially in our case 
    as our data has only 10% of the churned users & overall sample size is minimal

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series): Target variable.
    - scale (bool): StandardScaler before SMOTE.

    Returns:
    - X_resampled (np.ndarray or DataFrame): Balanced features.
    - y_resampled (np.ndarray or Series): Balanced target.
    """
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled

def split_data(df, target, test_size=0.2, stratify=True, scale=False):
    """
    Splits a DataFrame into train and test sets.

    Parameters:
    - df (pd.DataFrame): dataset
    - target (str): target column
    - test_size (float): Proportion of data for test set
    - stratify (bool): stratify flag
    - scale (bool): StandardScaler flag

    Return:
    - X_train, X_test, y_train, y_test (tuple): split datasets
    """
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Stratify if required
    stratify_col = y if stratify else None

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_col, random_state=42
    )

    # Optionally scale the data
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test=None, y_test=None, model_name='logistic', random_state=42):
    """
    Trains a classifier model based on the given model name.

    Supported models:
    - 'logistic'
    - 'random_forest'
    - 'svm'
    - 'xgboost'
    - 'lightgbm'

    Returns:
    - model: Trained model
    - metrics: Dictionary of evaluation metrics (if test data is provided)
    """
    
    # Select model
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=random_state)
    elif model_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Fit model
    model.fit(X_train, y_train)

    # Evaluate if test data is provided
    metrics = {}
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 'N/A'
        }

    return model, metrics


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def find_best_k_for_knn_imputer(df, columns_to_impute, k_range=range(1, 20), sample_frac=0.2, random_state=42):
    """
    Finds the best k for KNNImputer using the elbow method based on simulated missing values.

    Parameters:
    - df: DataFrame with original data
    - columns_to_impute: list of columns to perform KNN imputation on
    - k_range: range of k values to test (default: 1 to 20)
    - sample_frac: fraction of complete rows to artificially remove values from for testing
    - random_state: reproducibility

    Returns:
    - DataFrame of k values and corresponding MSE
    """
    df_copy = df.copy()

    # Remove rows with actual missing values in those columns
    df_no_na = df_copy.dropna(subset=columns_to_impute)

    # Sample a portion to simulate missingness
    sampled = df_no_na.sample(frac=sample_frac, random_state=random_state)
    df_with_simulated_na = df_copy.copy()
    df_with_simulated_na.loc[sampled.index, columns_to_impute] = np.nan

    errors = []

    for k in k_range:
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(df_with_simulated_na[columns_to_impute])
        mse = mean_squared_error(sampled[columns_to_impute], imputed[sampled.index])
        errors.append(mse)

    # Plot elbow
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), errors, marker='o')
    plt.title('Elbow Method for Optimal k in KNN Imputer')
    plt.xlabel('n_neighbors')
    plt.ylabel('MSE on Simulated Missing Data')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({'k': list(k_range), 'mse': errors})
