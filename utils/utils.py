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
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
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

def apply_smote(X, y, scale=False, tomek=False, sampling_strategy=0.3):
    """
    Applies SMOTE to balance the target classes, specially in our case 
    as our data has only 10% of the churned users & overall sample size is minimal

    Parameters:
    - X (pd.DataFrame): Feature dataset.
    - y (pd.Series): Target variable.
    - scale (bool): StandardScaler before SMOTE.
    - tomek flag: allows to use smoteTomek to oversample & reduce the noise from sample at the same time

    Returns:
    - X_resampled (np.ndarray or DataFrame): Balanced features.
    - y_resampled (np.ndarray or Series): Balanced target.
    """
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    if tomek:
        smote_tomek = SMOTETomek(random_state=42, sampling_strategy=sampling_strategy) # Adding sampling strategy
        X_resampled, y_resampled = smote_tomek.fit_resample(X_scaled, y)
    else:
        smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    return X_resampled, y_resampled

def split_data(df, target, test_size=0.2, stratify=True, scale=False, pca=False, 
               pca_variance=0.95, exclude_from_pca=None):
    """
    Split dataset and apply optional scaling + PCA to all columns EXCEPT those in `exclude_from_pca`.

    Parameters:
    - exclude_from_pca: list of columns to exclude from PCA and rejoin later

    Returns:
    - X_train_final, X_test_final, y_train, y_test, scaler, pca_model
    """
    if exclude_from_pca is None:
        exclude_from_pca = []

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split
    stratify_col = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_col, random_state=42
    )

    # Split data into PCA features and excluded features
    pca_cols = [col for col in X.columns if col not in exclude_from_pca]
    
    X_train_pca_input = X_train[pca_cols].copy()
    X_test_pca_input = X_test[pca_cols].copy()

    X_train_excluded = X_train[exclude_from_pca].copy()
    X_test_excluded = X_test[exclude_from_pca].copy()

    scaler = None
    pca_model = None

    # Optional scaling
    if scale:
        scaler = StandardScaler()
        X_train_pca_input = scaler.fit_transform(X_train_pca_input)
        X_test_pca_input = scaler.transform(X_test_pca_input)

    # Optional PCA
    if pca:
        pca_model = PCA(n_components=pca_variance)
        X_train_pca = pca_model.fit_transform(X_train_pca_input)
        X_test_pca = pca_model.transform(X_test_pca_input)

        pca_colnames = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
        X_train_pca = pd.DataFrame(X_train_pca, columns=pca_colnames, index=X_train.index)
        X_test_pca = pd.DataFrame(X_test_pca, columns=pca_colnames, index=X_test.index)
    else:
        X_train_pca = pd.DataFrame(X_train_pca_input, columns=pca_cols, index=X_train.index)
        X_test_pca = pd.DataFrame(X_test_pca_input, columns=pca_cols, index=X_test.index)

    # Combine PCA-transformed features with excluded original columns
    X_train_final = pd.concat([X_train_pca, X_train_excluded], axis=1)
    X_test_final = pd.concat([X_test_pca, X_test_excluded], axis=1)

    return X_train_final, X_test_final, y_train, y_test, scaler, pca_model

    """
    Splits a DataFrame into train and test sets with optional scaling and PCA.

    Parameters:
    - df (pd.DataFrame): Input dataset
    - target (str): Name of target column
    - test_size (float): Test set proportion
    - stratify (bool): Whether to stratify on the target column
    - scale (bool): Whether to standardize features
    - pca (bool): Whether to apply PCA
    - pca_variance (float): Variance to retain in PCA (default 0.95)

    Returns:
    - X_train, X_test, y_train, y_test (np.ndarrays)
    - Optionally: fitted scaler and PCA model for inverse transforms
    """

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Stratify flag
    stratify_col = y if stratify else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_col, random_state=42
    )

    # Optional: scale
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
    - 'decisionTree'

    Returns:
    - model: Trained model
    - metrics: Dictionary of evaluation metrics (if test data is provided)
    """
    
    # Select model
    if model_name == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight= 'balanced')
    elif model_name == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight= 'balanced')
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=random_state)
    elif model_name == 'xgboost':
        scale_pos_weight = len(y_train[y_train == 0])/len(y_train[y_train==1])
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, scale_pos_weight = scale_pos_weight)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(random_state=random_state)
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_name == 'grad_boost':
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Fit model
    model.fit(X_train, y_train)

    # Evaluate if test data is provided
    metrics = {}
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        #Adjust classification probability from 0.5 to 0.3 because cost of getting new customer could be higher than retaining (depends on business KPIs)
        y_pred_custom = (y_prob >= 0.3).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_custom),
            'precision': precision_score(y_test, y_pred_custom, zero_division=0),
            'recall': recall_score(y_test, y_pred_custom, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_custom, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 'N/A',
            'lift_metrics': calculate_lift_scores(y_test, y_prob)
        }

    return model, metrics

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

def tune_model_with_gridsearch(X, y, model_type='xgboost', cv=5, scoring='f1', verbose=1):
    """
    Performs GridSearchCV to optimize hyperparameters for XGBoost or LightGBM.

    Parameters:
    - X, y: Features and target
    - model_type: 'xgboost' or 'lightgbm'
    - cv: Number of cross-validation folds
    - scoring: Metric to optimize (e.g. 'f1', 'roc_auc')
    - verbose: Verbosity level

    Returns:
    - best_model: Trained model with best params
    - best_params: Dictionary of best parameters
    """
    if model_type == 'xgboost':
        model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'scale_pos_weight': [1, 3, 5]  # for imbalance
        }
    elif model_type == 'lightgbm':
        model = LGBMClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'class_weight': [None, 'balanced']  # for imbalance
        }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, class_weight= {0:1,1:9} )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'class_weight': [None, 'balanced']
        }
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']  # handles class imbalance
        }
    else:
        raise ValueError("Invalid model_type. Choose 'xgboost' or 'lightgbm'.")

    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, verbose=verbose, n_jobs=-1)
    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_

def calculate_lift_scores(y_true, y_proba, positive_label=1, cutoffs=[0.05, 0.10, 0.20]):
    """
    Calculate lift scores at different percentiles.
    
    Parameters:
    - y_true: Actual target values
    - y_proba: Predicted probabilities
    - positive_label: The label considered as 'positive' (e.g., 1 for churned)
    - cutoffs: List of top % cutoffs to evaluate (e.g., [0.05, 0.10, 0.20])
    
    Returns:
    - Dictionary: { 'lift_at_top_5%': value, ... }
    """
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False)

    lift_scores = {}
    base_rate = (df['y_true'] == positive_label).mean()

    for pct in cutoffs:
        cutoff = int(len(df) * pct)
        top_df = df.iloc[:cutoff]
        top_rate = (top_df['y_true'] == positive_label).mean()
        lift = top_rate / base_rate if base_rate > 0 else 0
        lift_scores[f'lift_at_top_{int(pct*100)}%'] = round(lift, 3)

    return lift_scores
