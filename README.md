
# Generalized Metadata & Profiling Engine

This project provides a **generalized metadata engine** for tabular datasets.
It automatically extracts dataset-level and column-level insights, computes inter-feature relationships, suggests preprocessing & feature engineering steps, and generates both **JSON metadata** and **HTML reports**.

---

## 🚀 Features

* **Dataset-level insights**

  * Number of rows/columns
  * Duplicate rows
  * Target distribution, imbalance ratio, suggested metrics

* **Column-level deep profiling**

  * Data types, roles (feature/id/timestamp)
  * Missing value ratio, unique counts, cardinality
  * Numeric stats (min, max, mean, std, skewness, kurtosis, outliers)
  * Categorical stats (top values, entropy, rare categories)
  * Preprocessing suggestions (imputation, encoding, scaling, transforms)

* **Inter-feature relationships**

  * Numeric vs numeric: Pearson correlation
  * Categorical vs categorical: Cramer’s V
  * Mixed: Mutual information

* **Feature importance estimation**

  * Mutual information (classification/regression)
  * Correlation with target

* **Feature interaction suggestions**

  * Polynomial (numeric × numeric)
  * Cross-feature (categorical × numeric / categorical × categorical)

* **High-correlation detection**

  * Detect redundant features (>0.85 correlation)

* **Reports**

  * JSON metadata (`generalized_mega_metadata.json`)
  * HTML metadata summary (`metadata_report.html`)
  * Full profiling report (`dataset_profile_report.html`)

---

## 📦 Installation

```bash
pip install pandas numpy ydata-profiling scipy scikit-learn featuretools json2html
```

---

## 🛠 Usage

1. **Run the script**

```bash
python metadata_engine.py
```

2. **Outputs generated**

   * `generalized_mega_metadata.json` → structured metadata in JSON
   * `metadata_report.html` → metadata converted into HTML view
   * `dataset_profile_report.html` → full profiling report from `ydata-profiling`

---

## 📂 Example

```python
df = pd.read_csv("/content/sample_data/california_housing_train.csv")
target_column = "median_house_value"
generalized_metadata = generate_generalized_metadata(df, target_column)
```

* Saves metadata as JSON
* Converts metadata into HTML
* Generates exploratory profiling report

---

## 📑 File Structure

```
.
├── metadata_engine.py              # Main script
├── generalized_mega_metadata.json  # JSON metadata output
├── metadata_report.html            # HTML metadata summary
├── dataset_profile_report.html     # Full profiling report
└── README.md                       # Project documentation
```

---

## ⚡ Future Improvements

* Add support for **time-series feature extraction** (lag features, rolling stats)
* Extend **feature importance** to include SHAP values
* Enable **real-time dataset profiling** (streaming data)

