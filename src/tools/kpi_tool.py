import pandas as pd
from src.tools.segmentation_tool import find_upgrade_candidates


def compute_business_kpis(df: pd.DataFrame) -> dict:
    """
    Computes headline business KPIs for the segmented customer base —
    used to power the Streamlit KPI dashboard (st.metric cards) and
    exposed to the agent as a tool for summary-style queries.
    """
    if df is None or df.empty or "Segment" not in df.columns:
        return {}

    bal_col = "avg_balance" if "avg_balance" in df.columns else "current_balance"
    spend_col = "total_spend" if "total_spend" in df.columns else None

    kpis = {
        "total_customers": int(len(df)),
        "priority_customers": int((df["Segment"] == "Priority").sum()),
        "regular_customers": int((df["Segment"] == "Regular").sum()),
        "dormant_customers": int((df["Segment"] == "Dormant").sum()),
    }

    if bal_col in df.columns:
        kpis["avg_balance"] = round(float(df[bal_col].mean()), 2)
    if spend_col and spend_col in df.columns:
        kpis["avg_total_spend"] = round(float(df[spend_col].mean()), 2)

    try:
        candidates = find_upgrade_candidates(df, top_n=len(df))
        kpis["cross_sell_potential"] = int(len(candidates))
    except Exception:
        kpis["cross_sell_potential"] = 0

    return kpis