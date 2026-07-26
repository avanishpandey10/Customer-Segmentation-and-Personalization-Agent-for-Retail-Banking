import pandas as pd


def get_segment_profiles(df: pd.DataFrame) -> dict:
    """Generates statistical profile summaries and rules for each customer segment."""
    if "Segment" not in df.columns:
        return {"error": "Dataset is not segmented yet."}

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    summary = df.groupby("Segment")[numeric_cols].mean().round(2).to_dict()
    counts = df["Segment"].value_counts().to_dict()

    return {
        "segment_counts": counts,
        "segment_averages": summary,
        "segmentation_rules": {
            "Priority": "High balance (>₹50k) or frequent high-value transactions.",
            "Regular": "Moderate account activity and consistent balances.",
            "Dormant": "Very low transaction count (<=1) or minimal account balance."
        }
    }


def explain_customer_segment(customer_row: dict, segment_df: pd.DataFrame = None) -> str:
    """
    Per-customer explainability: explains WHY this specific customer landed in
    their segment, tied to their actual values. When the full segmented
    dataframe is available, adds a percentile/ratio comparison against the
    Regular-segment average for extra concreteness (e.g. "3x the Regular
    average, top 8% by balance").
    """
    segment = customer_row.get("Segment", "Unknown")
    bal_col = "avg_balance" if "avg_balance" in customer_row else "current_balance"
    bal = customer_row.get(bal_col, "N/A")
    freq = customer_row.get("transaction_frequency", "N/A")

    base_reason = {
        "Priority": (
            f"average balance (₹{bal}) and/or transaction frequency ({freq}) exceed the "
            f"Priority thresholds (balance > ₹50,000, or balance > ₹20,000 with 10+ transactions)."
        ),
        "Regular": (
            f"balance (₹{bal}) and activity (frequency: {freq}) are moderate — active enough to "
            f"avoid Dormant, but not yet meeting the Priority thresholds."
        ),
        "Dormant": (
            f"transaction frequency ({freq}) is very low (≤1) or balance (₹{bal}) is under ₹1,000, "
            f"indicating an inactive account."
        ),
    }.get(segment, f"segment '{segment}' has no explanation rule defined.")

    extra = ""
    if segment_df is not None and "Segment" in segment_df.columns and bal_col in segment_df.columns:
        try:
            reg_avg_bal = segment_df.loc[segment_df["Segment"] == "Regular", bal_col].mean()
            if isinstance(bal, (int, float)) and reg_avg_bal and reg_avg_bal > 0:
                ratio = bal / reg_avg_bal
                percentile_below = (segment_df[bal_col] < bal).mean() * 100
                extra = (
                    f" For context: this balance is about {ratio:.1f}x the Regular-segment average, "
                    f"placing this customer in the top {100 - percentile_below:.0f}% of all customers by balance."
                )
        except Exception:
            pass

    return f"This customer is in **{segment}** because their {base_reason}{extra}"


def get_cross_sell_recommendations(segment_name: str, customer_row: dict = None) -> str:
    """
    Returns tailored banking product recommendations per customer persona.
    If a specific customer_row is provided, adds a secondary rule layer on
    top of the base segment strategy (e.g. high avg_transaction_size within
    Priority gets a different pitch than a Priority customer with modest
    transaction sizes).
    """
    segment_clean = str(segment_name).strip().capitalize()

    recommendations = {
        "Priority": (
            "🌟 **Priority Segment Strategy**\n\n"
            "• **Persona:** High-Net-Worth & Active Transactors\n"
            "• **Recommended Products:** Wealth Management Services, Premium Travel Credit Card, Tiered Fixed Deposits\n"
            "• **Actionable Pitch:** Provide dedicated relationship managers and exclusive cashback reward tiers."
        ),
        "Regular": (
            "💳 **Regular Segment Strategy**\n\n"
            "• **Persona:** Consistent Daily Banking Users\n"
            "• **Recommended Products:** Personal Loans, Auto Loans, Shopping Rewards Credit Card\n"
            "• **Actionable Pitch:** Encourage automated savings plans and offer pre-approved personal credit lines."
        ),
        "Dormant": (
            "🔄 **Dormant Segment Strategy**\n\n"
            "• **Persona:** Inactive / Low Balance Accounts\n"
            "• **Recommended Products:** Zero-Balance Digital Account, High-Yield Savings Promo, UPI Cashback Offers\n"
            "• **Actionable Pitch:** Send targeted re-engagement campaigns and waive account maintenance fees upon reaching a transaction threshold."
        )
    }

    base = recommendations.get(
        segment_clean,
        f"Unknown segment '{segment_name}'. Please choose from: Priority, Regular, or Dormant."
    )

    if customer_row and segment_clean in recommendations:
        avg_tx = customer_row.get("avg_transaction_size")
        if segment_clean == "Priority" and isinstance(avg_tx, (int, float)) and avg_tx > 10000:
            base += (
                "\n\n• **Secondary rule triggered:** Large average transaction size (₹{:.0f}) — "
                "also propose a Platinum/Metal Credit Card and forex-fee waivers for likely "
                "high-value or international spend.".format(avg_tx)
            )
        elif segment_clean == "Dormant":
            recency = customer_row.get("recency_days")
            if isinstance(recency, (int, float)) and recency > 90:
                base += (
                    f"\n\n• **Secondary rule triggered:** No activity in {int(recency)}+ days — "
                    f"prioritize a win-back SMS/email campaign before offering any new product."
                )

    return base