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

def get_cross_sell_recommendations(segment_name: str) -> dict:
    """Returns tailored banking product recommendations per customer persona."""
    recommendations = {
        "Priority": {
            "Persona": "High-Net-Worth & Active Transactors",
            "Products": ["Wealth Management Services", "Premium Travel Credit Card", "Fixed Deposits with Tiered Interest"],
            "Strategy": "Provide dedicated relationship managers and exclusive cashback reward tiers."
        },
        "Regular": {
            "Persona": "Consistent Daily Banking Users",
            "Products": ["Personal Loans", "Auto Loans", "Shopping Rewards Credit Card"],
            "Strategy": "Encourage automated savings plans and cross-sell pre-approved credit lines."
        },
        "Dormant": {
            "Persona": "Inactive / Low Balance Accounts",
            "Products": ["Zero-Balance Digital Account", "High-Yield Savings Promo", "UPI Cashback Offers"],
            "Strategy": "Send re-engagement campaigns, waive maintenance fees upon transaction threshold."
        }
    }
    return recommendations.get(segment_name, {"message": "Unknown segment."})