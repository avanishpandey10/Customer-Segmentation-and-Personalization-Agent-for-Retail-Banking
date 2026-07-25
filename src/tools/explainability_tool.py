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


def explain_customer_segment(customer_row: dict) -> str:
    """
    Per-customer explainability: generates a natural-language explanation of
    WHY this specific customer landed in their segment, tied to their actual
    balance/frequency values. Required by the brief: "Explainability (why a
    customer belongs to a segment)".
    """
    segment = customer_row.get("Segment", "Unknown")
    bal = customer_row.get("avg_balance", customer_row.get("current_balance", "N/A"))
    freq = customer_row.get("transaction_frequency", "N/A")

    reason_map = {
        "Priority": (
            f"This customer is in **Priority** because their average balance (₹{bal}) and/or "
            f"transaction frequency ({freq}) exceed the high-value thresholds "
            f"(balance > ₹50,000, or balance > ₹20,000 with more than 10 transactions)."
        ),
        "Regular": (
            f"This customer is in **Regular** because their balance (₹{bal}) and activity "
            f"(frequency: {freq}) fall in the moderate range — active enough to avoid Dormant, "
            f"but not yet meeting the Priority thresholds."
        ),
        "Dormant": (
            f"This customer is in **Dormant** because their transaction frequency ({freq}) is "
            f"very low (≤1) or their balance (₹{bal}) is under ₹1,000, indicating an inactive account."
        ),
    }
    return reason_map.get(segment, f"Segment '{segment}' has no explanation rule defined.")


def get_cross_sell_recommendations(segment_name: str) -> str:
    """Returns tailored banking product recommendations per customer persona."""
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

    return recommendations.get(
        segment_clean,
        f"Unknown segment '{segment_name}'. Please choose from: Priority, Regular, or Dormant."
    )