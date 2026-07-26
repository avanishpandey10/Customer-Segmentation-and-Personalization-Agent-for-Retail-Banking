import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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


def _compute_balance_volatility(df: pd.DataFrame, cust_col: str, bal_col: str) -> pd.Series:
    """Coefficient of variation of balance — high volatility may indicate irregular income/behavior."""
    if bal_col not in df.columns:
        return pd.Series(dtype=float)
    stats = df.groupby(cust_col)[bal_col].agg(["std", "mean"])
    stats["cv"] = (stats["std"] / stats["mean"].replace(0, np.nan)).fillna(0)
    return stats["cv"].rename("balance_volatility")


def _compute_weekend_ratio(df: pd.DataFrame, cust_col: str, date_col: str) -> pd.Series:
    """Proportion of transactions on weekends — proxy for personal vs business banking."""
    if date_col not in df.columns:
        return pd.Series(dtype=float)
    d = df[[cust_col, date_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return pd.Series(dtype=float)
    d["is_weekend"] = d[date_col].dt.dayofweek.isin([5, 6]).astype(int)
    return d.groupby(cust_col)["is_weekend"].mean().rename("weekend_transaction_ratio")


def preprocess_and_aggregate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transactional data (e.g. bank_transactions.csv) into an
    aggregated customer-level dataset, including RFM-style features:
    Recency (recency_days), Frequency (transaction_frequency),
    Monetary (total_spend), plus spend_trend, balance_volatility,
    and weekend_transaction_ratio signals.
    """
    df.columns = [col.strip() for col in df.columns]

    cust_id_col = 'CustomerID' if 'CustomerID' in df.columns else df.columns[0]
    bal_col = 'CustAccountBalance' if 'CustAccountBalance' in df.columns else 'BALANCE'
    tx_amt_col = 'TransactionAmount (INR)' if 'TransactionAmount (INR)' in df.columns else 'PURCHASES'
    date_col = 'TransactionDate' if 'TransactionDate' in df.columns else None

    if tx_amt_col in df.columns and cust_id_col in df.columns:
        agg_dict = {}
        if bal_col in df.columns:
            agg_dict[bal_col] = ['last', 'mean', 'max', 'std']
        agg_dict[tx_amt_col] = ['count', 'sum', 'mean', 'std']

        customer_df = df.groupby(cust_id_col).agg(agg_dict)
        customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]

        rename_map = {
            f"{bal_col}_last": "current_balance",
            f"{bal_col}_mean": "avg_balance",
            f"{bal_col}_max": "max_balance",
            f"{bal_col}_std": "balance_std",
            f"{tx_amt_col}_count": "transaction_frequency",
            f"{tx_amt_col}_sum": "total_spend",
            f"{tx_amt_col}_mean": "avg_transaction_size",
            f"{tx_amt_col}_std": "transaction_size_std"
        }
        customer_df = customer_df.rename(columns={k: v for k, v in rename_map.items() if k in customer_df.columns})
        customer_df = customer_df.reset_index()

        # RFM-style extras
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

            try:
                weekend_ratio = _compute_weekend_ratio(df, cust_id_col, date_col)
                if not weekend_ratio.empty:
                    customer_df = customer_df.merge(weekend_ratio, left_on=cust_id_col, right_index=True, how="left")
            except Exception as e:
                logger.warning(f"Could not compute weekend_transaction_ratio: {e}")

        # Balance volatility
        try:
            bal_vol = _compute_balance_volatility(df, cust_id_col, bal_col)
            if not bal_vol.empty:
                customer_df = customer_df.merge(bal_vol, left_on=cust_id_col, right_index=True, how="left")
        except Exception as e:
            logger.warning(f"Could not compute balance_volatility: {e}")
    else:
        customer_df = df.copy()

    customer_df = customer_df.fillna(0)
    return customer_df


def select_most_important_features(df: pd.DataFrame, target_col: str = "Segment", top_k: int = 8) -> dict:
    """
    Performs feature selection using Random Forest importance.
    This addresses the hackathon brief's explicit requirement for
    "feature selection to derive the most definitive features of a customer."
    
    Returns a dict with ranked features and their importance scores.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found. Run segmentation first."}
    
    # Get numeric features only
    feature_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    # Remove identifier-like columns
    id_cols = [c for c in feature_cols if "id" in c.lower() or "cust" in c.lower()]
    feature_cols = [c for c in feature_cols if c not in id_cols and c != target_col]
    
    if len(feature_cols) < 3:
        return {"error": "Not enough numeric features for selection."}
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_k)
    
    return {
        "feature_importance": top_features.to_dict(orient="records"),
        "top_features_list": top_features['feature'].tolist(),
        "total_features_analyzed": len(feature_cols),
        "cumulative_importance_top_k": round(top_features['importance'].sum() * 100, 1),
        "interpretation": (
            f"The top {top_k} features capture "
            f"{top_features['importance'].sum()*100:.1f}% of the predictive power "
            f"for customer segmentation. These are the most definitive behavioral "
            f"and financial attributes that differentiate customer groups."
        )
    }