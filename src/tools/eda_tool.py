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


def generate_eda_summary(df: pd.DataFrame) -> str:
    """
    Produces a narrative, executive-friendly EDA summary — NOT a raw stats dump.
    This is the deliverable the hackathon brief calls a "Summary of EDA performed
    on the data" that a bank VP or marketing lead could actually read and act on.
    """
    if df is None or df.empty:
        return "No data available for EDA summary."

    total_rows = len(df)
    total_cols = len(df.columns)

    # Identify key columns heuristically
    cust_col = next(
        (c for c in df.columns if "cust" in c.lower() or "id" in c.lower()),
        None,
    )
    date_col = next(
        (c for c in df.columns if "date" in c.lower()),
        None,
    )
    amt_col = next(
        (c for c in df.columns if "amount" in c.lower() or "txn" in c.lower()),
        None,
    )
    bal_col = next(
        (c for c in df.columns if "balance" in c.lower()),
        None,
    )

    # Data quality
    missing_pct = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100
    if missing_pct == 0:
        quality = "excellent — no missing values detected"
    elif missing_pct < 1:
        quality = f"good — only {missing_pct:.1f}% of values are missing"
    elif missing_pct < 5:
        quality = (
            f"fair — {missing_pct:.1f}% missing values may need attention"
        )
    else:
        quality = (
            f"poor — {missing_pct:.1f}% missing values is significant; "
            f"imputation recommended"
        )

    # Date range
    date_info = "Date range could not be determined."
    if date_col:
        try:
            dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
            if dates.notna().any():
                date_info = (
                    f"Transactions span from "
                    f"{dates.min().strftime('%d %b %Y')} to "
                    f"{dates.max().strftime('%d %b %Y')}."
                )
        except Exception:
            pass

    # Unique customers
    customer_count = "unknown"
    if cust_col:
        customer_count = f"{df[cust_col].nunique():,} unique customers"

    # Transaction patterns
    tx_info = ""
    if amt_col and cust_col:
        avg_tx = df[amt_col].mean()
        total_tx_volume = df[amt_col].sum()
        tx_per_cust = df.groupby(cust_col).size().mean()
        tx_info = (
            f"Average transaction size: ₹{avg_tx:,.2f}. "
            f"Total transaction volume: ₹{total_tx_volume:,.0f}. "
            f"Customers average {tx_per_cust:.1f} transactions each."
        )

    # Balance profile
    bal_info = ""
    if bal_col:
        avg_bal = df[bal_col].mean()
        median_bal = df[bal_col].median()
        zero_bal_pct = (df[bal_col] == 0).mean() * 100
        bal_info = (
            f"Average account balance: ₹{avg_bal:,.0f} "
            f"(median: ₹{median_bal:,.0f}). "
            f"{zero_bal_pct:.1f}% of records show zero balance."
        )

    # Assemble the narrative summary
    summary_lines = [
        "## 📊 EDA Summary",
        "",
        f"**Dataset Overview:** {total_rows:,} records across {total_cols} "
        f"features, representing {customer_count}.",
        "",
        f"**Time Period:** {date_info}",
        "",
        f"**Transaction Patterns:** {tx_info}",
        "",
        f"**Balance Profile:** {bal_info}",
        "",
        f"**Data Quality:** The dataset is {quality}.",
        "",
        "**Key Columns Identified:**",
        f"- Customer identifier: `{cust_col or 'Not found'}`",
        f"- Transaction amounts: `{amt_col or 'Not found'}`",
        f"- Account balances: `{bal_col or 'Not found'}`",
        f"- Transaction dates: `{date_col or 'Not found'}`",
    ]

    return "\n".join(summary_lines)


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
        issues.append(
            "Dataset has fewer than 10 rows — results may not be statistically "
            "meaningful."
        )

    return issues