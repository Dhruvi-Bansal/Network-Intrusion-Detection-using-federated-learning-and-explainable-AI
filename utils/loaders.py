# =========================== utils/loaders.py ================================
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image

# -------------------------------------------------------------------
# GENERIC LOADERS
# -------------------------------------------------------------------

def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON {path}: {e}")
    return {}

def load_numpy(path):
    if os.path.exists(path):
        try:
            return np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading numpy file {path}: {e}")
    return None

def load_csv(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Error loading CSV {path}: {e}")
    return None

def load_image(path):
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return None

def load_keras_model(path):
    if not os.path.exists(path):
        print(f"Model path not found: {path}")
        return None
    try:
        if path.endswith(".h5"):
            return load_model(path, compile=False)
        elif path.endswith(".pkl"):
            return joblib.load(path)
        else:
            print(f"Unknown model type: {path}")
            return None
    except Exception as e:
        print(f"Could not load model: {e}")
        return None

# -------------------------------------------------------------------
# UNIVERSAL ASSET LOADER
# -------------------------------------------------------------------

def load_assets(folder_path):
    """
    Load all model, SHAP, and prediction-related files from a directory.
    Returns: (model, shap_vals, base_vals, features, X_sample, y_sample, preds, meta)
    """
    # --- MODEL ---
    model = None
    for fname in os.listdir(folder_path):
        if fname.endswith((".h5", ".pkl")):
            model = load_keras_model(os.path.join(folder_path, fname))
            break

    # --- SHAP VALUES ---
    shap_vals = None
    for fname in ["shap_values.npy", "shap_values_global.npy"]:
        shap_path = os.path.join(folder_path, fname)
        if os.path.exists(shap_path):
            shap_vals = load_numpy(shap_path)
            break

    # --- BASE VALUES ---
    base_vals = None
    for fname in ["base_values.npy", "expected_value.npy"]:
        base_path = os.path.join(folder_path, fname)
        if os.path.exists(base_path):
            base_vals = load_numpy(base_path)
            break

    # --- FEATURES ---
    features = None
    feature_path = os.path.join(folder_path, "features.json")
    if os.path.exists(feature_path):
        try:
            features = json.load(open(feature_path))
        except Exception as e:
            print(f"Error reading features.json: {e}")
    else:
        features = [f"feature_{i}" for i in range(16)]

    # --- X_SAMPLE ---
    X_sample = None
    for fname in ["X_sample.csv", "X_sample.npy"]:
        x_path = os.path.join(folder_path, fname)
        if os.path.exists(x_path):
            X_sample = load_csv(x_path) if x_path.endswith(".csv") else load_numpy(x_path)
            break

    # --- Y_SAMPLE ---
    y_sample = None
    for fname in ["y_sample.csv", "y_sample.npy"]:
        y_path = os.path.join(folder_path, fname)
        if os.path.exists(y_path):
            y_sample = load_csv(y_path)
            if isinstance(y_sample, pd.DataFrame):
                y_sample = y_sample.squeeze().values
            elif isinstance(y_sample, pd.Series):
                y_sample = y_sample.values
            break

    # --- PREDICTIONS ---
    preds = None
    for fname in ["predictions.csv", "predictions.npy"]:
        p_path = os.path.join(folder_path, fname)
        if os.path.exists(p_path):
            preds = load_csv(p_path)
            if isinstance(preds, pd.DataFrame):
                preds = preds.squeeze().values
            elif isinstance(preds, pd.Series):
                preds = preds.values
            break

    # --- METADATA ---
    meta = {}
    for fname in ["results.json", "privacy.json", "federated_accuracy.json"]:
        path = os.path.join(folder_path, fname)
        if os.path.exists(path):
            try:
                meta.update(load_json(path))
            except Exception as e:
                print(f"Error reading {fname}: {e}")

    return model, shap_vals, base_vals, features, X_sample, y_sample, preds, meta
