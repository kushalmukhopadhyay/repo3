import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import skew, kurtosis, entropy
import json

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="MetaGlimpse", page_icon="🧬", layout="wide")

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def compute_feature_weight(col, y_encoded, target_type):
    # (Same logic as before, kept lightweight)
    if pd.api.types.is_numeric_dtype(col):
        if target_type == "regression":
            return abs(col.corr(y_encoded))
        return float(mutual_info_classif(col.fillna(0).to_numpy().reshape(-1,1), y_encoded, discrete_features=False)[0])
    else:
        le = LabelEncoder()
        try:
            encoded = le.fit_transform(col.fillna("missing"))
            if target_type == "classification":
                return float(mutual_info_classif(encoded.reshape(-1,1), y_encoded, discrete_features=True)[0])
            return float(mutual_info_regression(encoded.reshape(-1,1), y_encoded)[0])
        except:
            return 0.0

# -----------------------------------------------------------------------------
# 3. HEAVY LIFTING (LAZY LOADED)
# -----------------------------------------------------------------------------
def generate_profile_report(df):
    # IMPORT HERE instead of at the top
    # This prevents the app from crashing on startup!
    with st.spinner("Loading Profiling Engine... (This happens only once)"):
        from ydata_profiling import ProfileReport
        pr = ProfileReport(df, minimal=True)
        return pr

def generate_metadata(df, target_column):
    # (Your existing metadata logic)
    metadata = {
        "dataset": {"rows": df.shape[0], "columns": df.shape[1], "duplicates": int(df.duplicated().sum())},
        "target": {}, "columns": {}, "pipeline_hints": []
    }
    
    y = df[target_column]
    target_type = "classification" if y.nunique() < 20 or y.dtype=='object' else "regression"
    y_encoded = y.copy()
    if target_type == "classification" and y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    
    metadata["target"] = {"name": target_column, "type": target_type}

    for col in df.columns:
        if col == target_column: continue
        col_data = df[col]
        col_meta = {
            "dtype": str(col_data.dtype),
            "missing_pct": round(col_data.isna().mean() * 100, 2),
            "cardinality": col_data.nunique()
        }
        col_meta["importance_weight"] = round(compute_feature_weight(col_data, y_encoded, target_type), 4)
        metadata["columns"][col] = col_meta

    return metadata

# -----------------------------------------------------------------------------
# 4. UI STRUCTURE
# -----------------------------------------------------------------------------
st.title("🧬 MetaGlimpse")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    target_col = st.selectbox("Select Target", df.columns)
    
    if st.button("Launch Analysis"):
        
        # 1. Light Tasks First
        meta = generate_metadata(df, target_col)
        st.json(meta, expanded=False)
        
        # 2. Heavy Tasks Last (Lazy Import)
        from streamlit_pandas_profiling import st_profile_report
        pr = generate_profile_report(df)
        st_profile_report(pr)

elif uploaded_file is None:
    st.info("Upload a CSV to start.")
