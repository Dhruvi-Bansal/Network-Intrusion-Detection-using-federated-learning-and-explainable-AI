# =========================== utils/shap_helpers.py ================================
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------- GLOBAL FEATURE IMPORTANCE ------------------------------------
def plot_global_shap(shap_vals, features):
    shap_vals = np.array(shap_vals)
    
    # 1. Handle 3D array (Samples, Features, Classes)
    if shap_vals.ndim == 3:
        # Take the absolute mean across classes (axis=2) to get (Samples, Features)
        abs_shap = np.mean(np.abs(shap_vals), axis=2) 
        # Then, take the mean across all samples (axis=0) to get (Features,)
        mean_abs = np.mean(abs_shap, axis=0).flatten()
    elif shap_vals.ndim == 2:
        # For 2D array (Samples, Features), just take the mean across samples
        mean_abs = np.mean(np.abs(shap_vals), axis=0).flatten()
    else:
        # Handle other unexpected dimensions if needed, or raise error
        st.error(f"Unexpected SHAP values dimension: {shap_vals.ndim}")
        return px.bar()


    if len(mean_abs) != len(features):
        st.warning("Feature count mismatch. Using feature indices.")
        features = [str(i) for i in range(len(mean_abs))]

    df = pd.DataFrame({
        "Feature": features,
        "Mean |SHAP|": mean_abs
    }).sort_values("Mean |SHAP|", ascending=False)

    fig = px.bar(
        df,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Global SHAP Feature Importance",
        color="Mean |SHAP|",
        color_continuous_scale="RdBu"
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    return fig

# -------- LOCAL FEATURE IMPACT ----------------------------------------
def plot_sample_shap(shap_vals, X, features, sample_index=0):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features[:X.shape[1]])
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if len(features) != X.shape[1]:
        features = X.columns.tolist()

    shap_vals = np.array(shap_vals)
    
    # 2. Handle 3D array (Samples, Features, Classes) -> reduce to (Samples, Features)
    if shap_vals.ndim == 3:
        # Average across the classes (axis=2)
        shap_vals = shap_vals.mean(axis=2) 
    # If 1D, reshape to (1, -1) for consistency (though unlikely here)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    # Now shap_vals is (Samples, Features) size, ready for indexing
    sample_index = int(sample_index)
    
    shap_row = shap_vals[sample_index] 
    feat_vals = X.iloc[sample_index]

    min_len = min(len(features), len(feat_vals), len(shap_row))
    features = list(features[:min_len])
    shap_row = shap_row[:min_len]
    feat_vals = feat_vals.iloc[:min_len]

    df = pd.DataFrame({
        "Feature": features,
        "Feature Value": feat_vals.values,
        "SHAP Value": shap_row
    }).sort_values("SHAP Value", key=lambda x: abs(x), ascending=True)

    fig = px.bar(
        df,
        x="SHAP Value",
        y="Feature",
        color="SHAP Value",
        color_continuous_scale="RdBu",
        title=f"Local SHAP Explanation for Sample #{sample_index}",
        orientation="h",
        height=600
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    
    # Removed text explanation logic from here as it's better placed in app.py

    return fig