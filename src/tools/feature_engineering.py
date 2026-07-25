import pandas as pd
import numpy as np

def preprocess_and_aggregate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transactional data (e.g. bank_transactions.csv)
    into a aggregated customer-level dataset.
    """
    # Standardize column names
    df.columns = [col.strip() for col in df.columns]
    
    # Check for required columns
    cust_id_col = 'CustomerID' if 'CustomerID' in df.columns else df.columns[0]
    bal_col = 'CustAccountBalance' if 'CustAccountBalance' in df.columns else 'BALANCE'
    tx_amt_col = 'TransactionAmount (INR)' if 'TransactionAmount (INR)' in df.columns else 'PURCHASES'
    
    # Aggregations
    if tx_amt_col in df.columns and cust_id_col in df.columns:
        agg_dict = {}
        if bal_col in df.columns:
            agg_dict[bal_col] = ['last', 'mean', 'max']
        agg_dict[tx_amt_col] = ['count', 'sum', 'mean']
        
        customer_df = df.groupby(cust_id_col).agg(agg_dict)
        customer_df.columns = ['_'.join(col).strip() for col in customer_df.columns.values]
        
        # Rename for simplicity
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
    else:
        customer_df = df.copy()

    # Handle missing values
    customer_df = customer_df.fillna(0)
    return customer_df