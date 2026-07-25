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