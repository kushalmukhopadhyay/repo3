# Install required libraries if needed:
# pip install pandas numpy ydata-profiling scipy scikit-learn featuretools

import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import skew, kurtosis, entropy, pearsonr
from itertools import combinations
import json
import json2html
from json2html import *

# ---------- Helper Functions ----------
def cramers_v(x, y):
    if x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    cm = pd.crosstab(x, y)
    chi2 = ((cm - cm.mean())**2 / cm.mean()).to_numpy().sum()
    n = cm.sum().sum()
    return np.sqrt(chi2 / (n * (min(cm.shape)-1))) if n>0 else 0

def compute_feature_weight(col, y_encoded, target_type):
    if pd.api.types.is_numeric_dtype(col):
        if target_type == "regression":
            return abs(col.corr(y_encoded))
        else:
            return float(mutual_info_classif(col.fillna(0).to_numpy().reshape(-1,1), y_encoded, discrete_features=False)[0])
    else:
        le = LabelEncoder()
        try:
            encoded = le.fit_transform(col.fillna("missing"))
            if target_type == "classification":
                return float(mutual_info_classif(encoded.reshape(-1,1), y_encoded, discrete_features=True)[0])
            else:
                return float(mutual_info_regression(encoded.reshape(-1,1), y_encoded)[0])
        except:
            return 0.0

def suggest_transform(col):
    """Suggest log transform for right-skewed numeric features"""
    if pd.api.types.is_numeric_dtype(col) and len(col.dropna())>1:
        if abs(skew(col.dropna())) > 1:
            return "log"
    return None

def detect_high_correlation(df, threshold=0.85):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j and corr_matrix.iloc[i,j] > threshold:
                high_corr.append({"features": [col1, col2], "correlation": corr_matrix.iloc[i,j]})
    return high_corr

# ---------- Main Generalized Metadata Engine ----------
def generate_generalized_metadata(df, target_column, top_n_interactions=5):
    metadata = {
        "dataset": {},
        "columns": {},
        "inter_feature_relationships": {},
        "feature_interactions": [],
        "pipeline_hints": {},
        "target": {}
    }

    # ---------- Dataset-level ----------
    metadata["dataset"] = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum())
    }

    # ---------- Target ----------
    y = df[target_column]
    target_type = "classification" if y.nunique() < 20 or y.dtype=='object' else "regression"
    y_encoded = y.copy()
    if target_type=="classification" and y.dtype=='object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

    metadata["target"] = {
        "name": target_column,
        "type": target_type,
        "distribution": y.value_counts().to_dict() if target_type=="classification" else None,
        "imbalance_ratio": round(y.value_counts().max()/y.value_counts().min(),2) if target_type=="classification" else None,
        "suggested_metrics": ["f1","roc_auc"] if target_type=="classification" else ["rmse","mae"]
    }

    # ---------- Column-level deep insights & generalized preprocessing ----------
    for col in df.columns:
        if col == target_column:
            continue
        data = df[col]
        col_meta = {}
        col_meta["dtype"] = str(data.dtype)
        col_meta["role"] = "feature"
        if "id" in col.lower():
            col_meta["role"] = "id"
        elif pd.api.types.is_datetime64_any_dtype(data):
            col_meta["role"] = "timestamp"
        col_meta["missing_pct"] = round(data.isna().sum()/len(df)*100,2)
        col_meta["unique_count"] = data.nunique()
        col_meta["cardinality_ratio"] = round(data.nunique()/len(df),4)

        # ---------- Numeric ----------
        if pd.api.types.is_numeric_dtype(data):
            clean_data = data.dropna()
            col_meta["stats"] = {
                "min": float(clean_data.min()) if not clean_data.empty else None,
                "max": float(clean_data.max()) if not clean_data.empty else None,
                "mean": float(clean_data.mean()) if not clean_data.empty else None,
                "median": float(clean_data.median()) if not clean_data.empty else None,
                "std": float(clean_data.std()) if not clean_data.empty else None,
                "skewness": float(skew(clean_data)) if len(clean_data)>1 else None,
                "kurtosis": float(kurtosis(clean_data)) if len(clean_data)>1 else None,
                "zero_ratio": round((clean_data==0).sum()/len(data),4) if not clean_data.empty else None,
                "negative_ratio": round((clean_data<0).sum()/len(data),4) if not clean_data.empty else None,
                "outlier_ratio": round(((clean_data - clean_data.mean()).abs() > 3*clean_data.std()).sum()/len(data),4) if len(clean_data)>1 else None
            }
            col_meta["preprocessing"] = {
                "imputation": "median" if col_meta["missing_pct"]>0 else "none",
                "scaling": "standard",
                "transform_suggestion": suggest_transform(data),
                "outlier_handling": "clip" if col_meta["stats"]["outlier_ratio"] > 0.01 else "none"
            }

        # ---------- Categorical ----------
        elif pd.api.types.is_categorical_dtype(data) or data.dtype=='object':
            value_counts = data.value_counts(normalize=True).head(5).to_dict()
            col_meta["top_values"] = {str(k): round(float(v),4) for k,v in value_counts.items()}
            col_meta["entropy"] = float(entropy(data.value_counts(normalize=True))) if data.nunique()>1 else 0
            col_meta["preprocessing"] = {
                "imputation": "most_frequent" if col_meta["missing_pct"]>0 else "none",
                "encoding": "target" if data.nunique()/len(df) > 0.1 else "one_hot",
                "rare_category_threshold": 0.01
            }

        # ---------- Datetime ----------
        elif pd.api.types.is_datetime64_any_dtype(data):
            col_meta["preprocessing"] = {"extract_features":["year","month","weekday","day","quarter","hour"], "cyclical_transform":True}

        # ---------- Boolean ----------
        elif data.dtype=="bool":
            col_meta["preprocessing"] = {"map_to_int": True}

        # ---------- Feature weightage ----------
        if col_meta["role"]=="feature":
            col_meta["weight"] = compute_feature_weight(data, y_encoded, target_type)

        metadata["columns"][col] = col_meta

    # ---------- Inter-feature relationships ----------
    features = [c for c in df.columns if c != target_column]
    for f1,f2 in combinations(features,2):
        data1, data2 = df[f1], df[f2]
        relation = {}
        if pd.api.types.is_numeric_dtype(data1) and pd.api.types.is_numeric_dtype(data2):
            corr,_ = pearsonr(data1.fillna(data1.mean()).to_numpy(), data2.fillna(data2.mean()).to_numpy())
            relation = {"type":"numeric","correlation":round(corr,3)}
        elif (data1.dtype=='object' or pd.api.types.is_categorical_dtype(data1)) and \
             (data2.dtype=='object' or pd.api.types.is_categorical_dtype(data2)):
            relation = {"type":"categorical","cramers_v":round(cramers_v(data1.fillna("missing"), data2.fillna("missing")),3)}
        else:
            # mixed
            numeric_data = data1 if pd.api.types.is_numeric_dtype(data1) else data2
            cat_data = data2 if pd.api.types.is_categorical_dtype(data2) else data1
            le_cat = LabelEncoder()
            try:
                encoded_cat = le_cat.fit_transform(cat_data.fillna("missing"))
                mi = mutual_info_classif(numeric_data.fillna(numeric_data.mean()).to_numpy().reshape(-1,1), encoded_cat, discrete_features=False)[0]
                relation = {"type":"mixed","mutual_info":round(float(mi),3)}
            except:
                relation = {"type":"mixed","mutual_info":0.0}
        metadata["inter_feature_relationships"][f"{f1}_{f2}"] = relation

    # ---------- Top feature interactions ----------
    sorted_features = sorted(metadata["columns"].items(), key=lambda x: -x[1].get("weight",0))
    top_features = [f[0] for f in sorted_features[:top_n_interactions]]
    for f1,f2 in combinations(top_features,2):
        interaction_type = "polynomial" if pd.api.types.is_numeric_dtype(df[f1]) and pd.api.types.is_numeric_dtype(df[f2]) else "cross_feature"
        metadata["feature_interactions"].append({"features":[f1,f2],"suggested_interaction":interaction_type})

    # ---------- High correlation hints ----------
    metadata["high_correlation_features"] = detect_high_correlation(df)

    # ---------- Profiling Report ----------
    profile = ProfileReport(df, title="Dataset Profile", explorative=True)
    profile.to_file("dataset_profile_report.html")

    return metadata

# ---------- Usage ----------
if __name__ == "__main__":
    df = pd.read_csv("/content/sample_data/california_housing_train.csv")
    target_column = "median_house_value"
    generalized_metadata = generate_generalized_metadata(df, target_column)

    with open("generalized_mega_metadata.json","w") as f:
        json.dump(generalized_metadata,f, indent=4)

    print("Generalized metadata JSON generated: generalized_mega_metadata.json")


with open("generalized_mega_metadata.json") as f:
    metadata = json.load(f)

html_content = json2html.convert(json = metadata)

with open("metadata_report.html", "w") as f:
    f.write(html_content)

print("Report saved to metadata_report.html")

