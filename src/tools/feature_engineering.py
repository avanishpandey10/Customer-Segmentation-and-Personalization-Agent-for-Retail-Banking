import pandas as pd
import numpy as np
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _compute_recency(df: pd.DataFrame, cust_col: str, date_col: str) -> pd.Series:
    """Days since each customer's most recent transaction, relative to the latest date in the data."""
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    if dates.isna().all():
        return pd.Series(dtype=float)
    max_date = dates.max()
    last_txn = dates.groupby(df[cust_col]).max()
    return (max_date - last_txn).dt.days.rename("recency_days")


def _compute_spend_trend(df: pd.DataFrame, cust_col: str, date_col: str, amt_col: str) -> pd.Series:
    """
    +1 if a customer's spending is trending up (2nd half of their history > 1st half),
    -1 if trending down, 0 if flat/insufficient data. Cheap proxy for behavioral trend.
    """
    d = df[[cust_col, date_col, amt_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values([cust_col, date_col])
    if d.empty:
        return pd.Series(dtype=float)

    d["rank"] = d.groupby(cust_col).cumcount()
    counts = d.groupby(cust_col)[amt_col].transform("count")
    d["half"] = np.where(d["rank"] < counts / 2, "first", "second")

    pivot = d.groupby([cust_col, "half"])[amt_col].mean().unstack("half")
    first = pivot.get("first", pd.Series(0, index=pivot.index))
    second = pivot.get("second", pd.Series(0, index=pivot.index))
    trend = np.sign(second.fillna(0) - first.fillna(0))
    return trend.rename("spend_trend")


def preprocess_and_aggregate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transactional data (e.g. bank_transactions.csv) into an
    aggregated customer-level dataset, including RFM-style features:
    Recency (recency_days), Frequency (transaction_frequency),
    Monetary (total_spend), plus a lightweight spend_trend signal.
    """
    df.columns = [col.strip() for col in df.columns]

    cust_id_col = 'CustomerID' if 'CustomerID' in df.columns else df.columns[0]
    bal_col = 'CustAccountBalance' if 'CustAccountBalance' in df.columns else 'BALANCE'
    tx_amt_col = 'TransactionAmount (INR)' if 'TransactionAmount (INR)' in df.columns else 'PURCHASES'
    date_col = 'TransactionDate' if 'TransactionDate' in df.columns else None

    if tx_amt_col in df.columns and cust_id_col in df.columns:
        agg_dict = {}
        if bal_col in df.columns:
            agg_dict[bal_col] = ['last', 'mean', 'max']
        agg_dict[tx_amt_col] = ['count', 'sum', 'mean']

        customer_df = df.groupby(cust_id_col).agg(agg_dict)
        customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]

        rename_map = {
            f"{bal_col}_last": "current_balance",
            f"{bal_col}_mean": "avg_balance",
            f"{bal_col}_max": "max_balance",
            f"{tx_amt_col}_count": "transaction_frequency",
            f"{tx_amt_col}_sum": "total_spend",
            f"{tx_amt_col}_mean": "avg_transaction_size"
        }
        customer_df = customer_df.rename(columns={k: v for k, v in rename_map.items() if k in customer_df.columns})
        customer_df = customer_df.reset_index()

        # RFM-style extras (best-effort — skipped gracefully if dates are unusable)
        if date_col:
            try:
                recency = _compute_recency(df, cust_id_col, date_col)
                if not recency.empty:
                    customer_df = customer_df.merge(recency, left_on=cust_id_col, right_index=True, how="left")
            except Exception as e:
                logger.warning(f"Could not compute recency_days: {e}")

            try:
                trend = _compute_spend_trend(df, cust_id_col, date_col, tx_amt_col)
                if not trend.empty:
                    customer_df = customer_df.merge(trend, left_on=cust_id_col, right_index=True, how="left")
            except Exception as e:
                logger.warning(f"Could not compute spend_trend: {e}")
    else:
        customer_df = df.copy()

    customer_df = customer_df.fillna(0)
    return customer_df