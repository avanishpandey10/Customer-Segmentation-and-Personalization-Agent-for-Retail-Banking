import pandas as pd
import numpy as np


def run_eda(df: pd.DataFrame) -> dict:
    """Performs automated exploratory data analysis on raw customer data."""
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
    }
    return summary


def query_eda_metrics(df: pd.DataFrame, metric_type: str, column_name: str = None) -> dict:
    """Calculates specific EDA metrics based on user request."""
    if metric_type == "missing_values":
        return {"missing_values": df.isnull().sum().to_dict()}
    elif metric_type == "summary_stats" and column_name in df.columns:
        return {"stats": df[column_name].describe().to_dict()}
    elif metric_type == "correlation":
        num_df = df.select_dtypes(include=[np.number])
        return {"correlation_matrix": num_df.corr().round(3).to_dict()}
    else:
        return {"error": "Invalid metric type or column name."}


def validate_dataset(df: pd.DataFrame) -> list:
    """
    Basic data validation. Returns a list of human-readable warnings/errors;
    empty list means the dataset looks usable.
    """
    issues = []
    if df is None or df.empty:
        issues.append("Dataset is empty.")
        return issues

    expected_any_of = [
        {"CustomerID", "CustAccountBalance", "TransactionAmount (INR)"},
    ]
    cols = set(df.columns)
    if not any(expected.issubset(cols) for expected in expected_any_of):
        issues.append(
            "Could not find the expected columns (CustomerID, CustAccountBalance, "
            "TransactionAmount (INR)). Segmentation quality may be degraded — "
            "check your column names."
        )

    if len(df) < 10:
        issues.append("Dataset has fewer than 10 rows — results may not be statistically meaningful.")

    return issues