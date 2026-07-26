import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools.feature_engineering import preprocess_and_aggregate_customer_data
from src.tools.segmentation_tool import segment_customers, find_upgrade_candidates
from src.tools.explainability_tool import get_segment_profiles, get_cross_sell_recommendations, explain_customer_segment
from src.tools.kpi_tool import compute_business_kpis


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "TransactionID": [f"T{i}" for i in range(1, 11)],
        "CustomerID": ["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C4", "C5", "C5"],
        "CustAccountBalance": [60000, 61000, 15000, 15500, 500, 400, 30000, 31000, 900, 950],
        "TransactionDate": ["02-08-2016"] * 10,
        "TransactionTime": [100000] * 10,
        "TransactionAmount (INR)": [500, 600, 200, 250, 10, 5, 15000, 300, 50, 60],
    })


def test_preprocessing_produces_expected_columns(raw_df):
    result = preprocess_and_aggregate_customer_data(raw_df)
    assert "CustomerID" in result.columns
    assert "avg_balance" in result.columns
    assert "transaction_frequency" in result.columns
    assert len(result) == 5  # 5 unique customers


def test_segmentation_rules_assigns_all_customers(raw_df):
    processed = preprocess_and_aggregate_customer_data(raw_df)
    segmented, metrics = segment_customers(processed, method="rules")
    assert "Segment" in segmented.columns
    assert set(segmented["Segment"].unique()).issubset({"Priority", "Regular", "Dormant"})
    assert metrics["method"] == "rules"
    assert "segment_distribution" in metrics


def test_segmentation_kmeans_runs(raw_df):
    processed = preprocess_and_aggregate_customer_data(raw_df)
    segmented, metrics = segment_customers(processed, method="kmeans", n_clusters=3)
    assert "Segment" in segmented.columns
    assert metrics["method"] == "kmeans"


def test_get_segment_profiles(raw_df):
    processed = preprocess_and_aggregate_customer_data(raw_df)
    segmented, _ = segment_customers(processed, method="rules")
    profiles = get_segment_profiles(segmented)
    assert "segment_counts" in profiles
    assert "segmentation_rules" in profiles


def test_get_cross_sell_recommendations_known_segment():
    output = get_cross_sell_recommendations("Priority")
    assert "Priority" in output
    assert "Wealth Management" in output


def test_get_cross_sell_recommendations_unknown_segment():
    output = get_cross_sell_recommendations("NotASegment")
    assert "Unknown segment" in output


def test_find_upgrade_candidates(raw_df):
    processed = preprocess_and_aggregate_customer_data(raw_df)
    segmented, _ = segment_customers(processed, method="rules")
    candidates = find_upgrade_candidates(segmented)
    # Should not error even if there are zero Regular customers in this tiny fixture
    assert isinstance(candidates, pd.DataFrame)


def test_explain_customer_segment_has_reason():
    row = {"Segment": "Priority", "avg_balance": 60000, "transaction_frequency": 12}
    explanation = explain_customer_segment(row)
    assert "Priority" in explanation


def test_compute_business_kpis(raw_df):
    processed = preprocess_and_aggregate_customer_data(raw_df)
    segmented, _ = segment_customers(processed, method="rules")
    kpis = compute_business_kpis(segmented)
    assert kpis["total_customers"] == 5
    assert "priority_customers" in kpis