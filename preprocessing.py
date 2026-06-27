import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


#CONFIG
ID_COL = 'customerID'          # unique id, no predictive value
TARGET_COL = 'Churn'           # Yes/No, becomes the marketing segment label
RANDOM_STATE = 42

# Yes/No (or Male/Female) columns that map cleanly to 0/1.
BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

# Continuous columns that need scaling.
NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Multi level categoricals that get one hot encoded.
MULTI_CAT_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod',
]


def load_data(filepath: str) -> pd.DataFrame:
    """Read the raw CSV and fail loudly if the path is wrong."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[Load] {df.shape[0]:,} rows x {df.shape[1]} columns from '{filepath}'.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the known data quality issues.

    TotalCharges arrives as text because 11 brand new customers (tenure 0)
    have a blank value. We coerce to numeric and set those blanks to 0,
    since a customer with zero months has not been charged a total yet.
    """
    df = df.copy()

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n_blank = int(df['TotalCharges'].isna().sum())
    if n_blank:
        df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = 0.0
        print(f"[Clean] Filled {n_blank} blank TotalCharges (all tenure 0) with 0.")

    df = df.drop(columns=[ID_COL], errors='ignore')
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Audit for nulls so a future data refresh cannot pass NaNs silently."""
    missing = df.isnull().sum()
    total = int(missing.sum())
    if total == 0:
        print(f"[Missing] None across all {df.shape[1]} columns.")
    else:
        print(f"[Missing] {total} values detected:")
        print(missing[missing > 0].to_string())
    return missing


def encode_features(df: pd.DataFrame) -> tuple:
    """
    Turn the raw frame into a numeric feature matrix and a 0/1 target.

    Binary columns map to 0/1, the target Yes/No maps to 1/0, and the
    multi level categoricals get one hot encoded with the first level
    dropped to avoid a redundant column.
    """
    df = df.copy()

    # Target first: churn Yes is the segment we want to flag, so it is 1.
    y = df[TARGET_COL].map({'Yes': 1, 'No': 0}).astype(int)
    df = df.drop(columns=[TARGET_COL])

    # Simple Yes/No and gender columns.
    for col in BINARY_COLS:
        if col == 'gender':
            df[col] = df[col].map({'Male': 1, 'Female': 0})
        else:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One hot the rest. drop_first removes one level per column.
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)

    # get_dummies returns bool columns, cast everything to float for Keras.
    X = df.astype(float)

    print(f"[Encode] Feature matrix is {X.shape[1]} columns after encoding.")
    print(f"[Target] Churn rate: {y.mean()*100:.2f}% positive ({int(y.sum())} of {len(y)}).")
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.15, val_size: float = 0.15) -> tuple:
    """
    Stratified 70/15/15 train/val/test split.

    Stratifying on churn keeps the positive rate steady in every split,
    which matters because the classes are imbalanced.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp,
        random_state=RANDOM_STATE
    )
    print(f"[Split] Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_features(X_train, X_val, X_test,
                       numeric_cols: list = NUMERIC_COLS,
                       scaler_save_path: str = 'outputs/scaler.pkl') -> tuple:
    """
    Standardize the continuous columns only.

    The one hot and binary columns are already 0/1, so scaling them adds
    nothing. We fit on training data alone, then reuse that scaler so no
    test statistics leak backwards.
    """
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    joblib.dump(scaler, scaler_save_path)
    print(f"[Scale] StandardScaler fit on {len(numeric_cols)} numeric columns, saved to '{scaler_save_path}'.")
    return X_train.values, X_val.values, X_test.values, scaler


def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Build balanced class weights for the imbalanced target.

    Without this the model can score about 73% accuracy by always
    predicting No churn, which is useless for finding who to market to.
    """
    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[Weights] Class weights: {weight_dict}")
    return weight_dict


def run_preprocessing(filepath: str,
                      scaler_save_path: str = 'outputs/scaler.pkl') -> dict:
    """Run every step in order and hand back a dict the rest of the code unpacks."""
    print("\n" + "-" * 55)
    print("  PREPROCESSING")
    print("-" * 55)

    df = load_data(filepath)
    df = clean_data(df)
    check_missing_values(df)

    X, y = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_s, X_val_s, X_test_s, scaler = normalize_features(
        X_train, X_val, X_test, scaler_save_path=scaler_save_path
    )

    class_weights = get_class_weights(y_train.values)

    print("-" * 55)
    print("  Preprocessing done.\n")

    return {
        'X_train': X_train_s, 'X_val': X_val_s, 'X_test': X_test_s,
        'y_train': y_train.values, 'y_val': y_val.values, 'y_test': y_test.values,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'class_weights': class_weights,
    }
