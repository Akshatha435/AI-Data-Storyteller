
# eda.py
"""
EDA utilities for AI Data Storyteller project
"""
import pandas as pd
import numpy as np
from typing import Dict

# ---------------- Load CSV ----------------
def load_csv(file) -> pd.DataFrame:
    """Load a CSV file into DataFrame. Accepts file path or file-like (Streamlit upload)."""
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin1")


# ---------------- Detect column types ----------------
def detect_column_types(df: pd.DataFrame, numeric_thresh: float = 0.9) -> Dict[str, list]:
    """
    Simple classification of columns into 'numerical' and 'categorical'.
    If >= numeric_thresh fraction of non-null values can be coerced to numeric, treat as numerical.
    """
    numerical = []
    categorical = []
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            numerical.append(col)
            continue
        non_null = ser.dropna()
        if non_null.empty:
            categorical.append(col)
            continue
        coerced = pd.to_numeric(non_null, errors="coerce")
        frac_numeric = coerced.notna().sum() / len(non_null)
        if frac_numeric >= numeric_thresh:
            numerical.append(col)
        else:
            categorical.append(col)
    return {"numerical": numerical, "categorical": categorical}


# ---------------- Fill missing (helper) ----------------
def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Return a copy of df with missing values filled according to strategy.
    strategy: 'mean', 'median', 'mode', 'zero', 'unknown', 'drop'
    - 'mean'/'median' apply to numeric columns
    - 'mode' uses the mode for any column
    - 'zero' numeric->0, categorical->'Unknown'
    - 'unknown' numeric->median, categorical->'Unknown'
    - 'drop' drops rows with any NA
    """
    df_copy = df.copy()
    if strategy == "drop":
        return df_copy.dropna()

    for col in df_copy.columns:
        if not df_copy[col].isnull().any():
            continue
        if strategy == "mode":
            try:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode(dropna=True).iloc[0])
            except Exception:
                df_copy[col] = df_copy[col].fillna("Unknown")
        elif strategy in ("mean", "median"):
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                if strategy == "mean":
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
                else:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode(dropna=True).iloc[0] if not df_copy[col].mode().empty else "Unknown")
        elif strategy == "zero":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(0)
            else:
                df_copy[col] = df_copy[col].fillna("Unknown")
        elif strategy == "unknown":
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            else:
                df_copy[col] = df_copy[col].fillna("Unknown")
        else:
            try:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode(dropna=True).iloc[0])
            except Exception:
                df_copy[col] = df_copy[col].fillna("Unknown")
    return df_copy


# ---------------- Run EDA (report only) ----------------
def run_eda(df: pd.DataFrame) -> dict:
    """
    Run a compact EDA and return a dictionary with keys used by the app:
      - 'types': {'numerical': [...], 'categorical': [...]}
      - 'summary': includes 'shape' and 'missing_values' (dict) and 'duplicates'
      - 'correlations': dict-of-dicts for numeric columns (so pd.DataFrame(...) works)
    IMPORTANT: This function does NOT mutate the input or impute missing values;
    it only reports current state.
    """
    out = {}
    out["summary"] = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }

    types = detect_column_types(df)
    out["types"] = types

    try:
        out["summary"]["dtypes"] = df.dtypes.astype(str).to_dict()
    except Exception:
        out["summary"]["dtypes"] = {}

    try:
        out["numeric_stats"] = df.describe(include=[np.number]).to_dict()
    except Exception:
        out["numeric_stats"] = {}

    cat_counts = {}
    for col in types.get("categorical", []):
        try:
            cat_counts[col] = df[col].value_counts(dropna=False).head(10).to_dict()
        except Exception:
            cat_counts[col] = {}
    out["categorical_counts"] = cat_counts

    numerical = types.get("numerical", [])
    if numerical and len(numerical) >= 1:
        try:
            corr_df = df[numerical].corr(method="pearson")
            correlations = corr_df.fillna(0).to_dict()
        except Exception:
            correlations = {c: {c2: 0.0 for c2 in numerical} for c in numerical}
    else:
        correlations = {c: {c2: 0.0 for c2 in numerical} for c in numerical}
    out["correlations"] = correlations

    try:
        out["summary"]["head"] = df.head(5).to_dict(orient="list")
    except Exception:
        out["summary"]["head"] = {}

    return out


# ---------------- Long Insights ----------------
def generate_long_insights(eda_result: dict) -> str:
    lines = []
    summary = eda_result.get("summary", {})
    types = eda_result.get("types", {})
    numeric_stats = eda_result.get("numeric_stats", {})
    categorical_counts = eda_result.get("categorical_counts", {})
    correlations = eda_result.get("correlations", {})

    shape = summary.get("shape", (None, None))
    lines.append(
        f"The dataset contains {shape[0]} rows and {shape[1]} columns. "
        "This report describes the dataset as currently loaded; no imputations were applied here—only reporting."
    )

    missing = summary.get("missing_values", {})
    nonzero = {k: v for k, v in missing.items() if v and v > 0}
    if nonzero:
        top_miss = sorted(nonzero.items(), key=lambda x: x[1], reverse=True)[:6]
        lines.append("Columns with the most missing values: " +
                     ", ".join([f"{k} ({v})" for k, v in top_miss]) + ".")
    else:
        lines.append("There are no missing values reported.")

    num_cols = types.get("numerical", [])
    if num_cols:
        lines.append(f"There are {len(num_cols)} numeric columns, including: "
                     + ", ".join(num_cols[:8]) + (", ..." if len(num_cols) > 8 else "."))

    cat_cols = types.get("categorical", [])
    if cat_cols:
        lines.append(f"There are {len(cat_cols)} categorical columns. "
                     "Key examples include: " + ", ".join(cat_cols[:8]) +
                     (", ..." if len(cat_cols) > 8 else "."))

    if correlations:
        pairs = []
        for a, sub in correlations.items():
            for b, v in sub.items():
                if a != b and v is not None:
                    pairs.append((a, b, v))
        seen = set()
        uniq = []
        for a, b, v in pairs:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                uniq.append((a, b, v))
        uniq_sorted = sorted(uniq, key=lambda x: abs(x[2]), reverse=True)[:5]
        if uniq_sorted:
            lines.append("Top correlations include: " +
                         "; ".join([f"{a}-{b} (r={v:.2f})" for a, b, v in uniq_sorted]) + ".")

    lines.append("Recommendations: if you wish to impute values, call `fill_missing` with an appropriate strategy or use the UI options. "
                 "Consider scaling numeric columns and grouping rare categorical values before downstream modeling.")

    return "\n\n".join(lines)
