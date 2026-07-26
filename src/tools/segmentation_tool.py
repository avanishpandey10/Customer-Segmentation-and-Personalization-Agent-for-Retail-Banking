import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.logger_setup import get_logger

logger = get_logger(__name__)

# Columns that actually define "priority-ness" per the hackathon brief
# (balance being maintained + frequency of transactions), used to label
# clusters instead of trusting arbitrary column order.
RANKING_COLS_PRIORITY = ["avg_balance", "current_balance", "transaction_frequency"]

PRIORITY_BALANCE_THRESHOLD = 50000
PRIORITY_BALANCE_WITH_FREQ = 20000
PRIORITY_FREQ_THRESHOLD = 10


def _rank_and_label_clusters(
    df_segmented: pd.DataFrame,
    num_cols,
    scaled_data: np.ndarray,
    cluster_col: str = "Cluster",
):
    """Ranks clusters by a balance+frequency composite score and maps them
    to Priority/Regular/Dormant."""
    ranking_cols = [c for c in RANKING_COLS_PRIORITY if c in num_cols]
    if not ranking_cols:
        ranking_cols = [num_cols[0]]
    ranking_idxs = [list(num_cols).index(c) for c in ranking_cols]

    cluster_scores = (
        pd.Series(scaled_data[:, ranking_idxs].mean(axis=1), index=df_segmented.index)
        .groupby(df_segmented[cluster_col])
        .mean()
        .sort_values()
    )
    label_map = {
        cluster_scores.index[0]: "Dormant",
        cluster_scores.index[-1]: "Priority",
    }
    for idx in cluster_scores.index:
        if idx not in label_map:
            label_map[idx] = "Regular"
    return label_map, ranking_cols


def _cluster_eval_metrics(scaled_data: np.ndarray, labels: np.ndarray) -> dict:
    """Computes silhouette / Davies-Bouldin / Calinski-Harabasz, subsampled
    for speed on large data."""
    metrics = {}
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        metrics[
            "evaluation_note"
        ] = "Fewer than 2 clusters found — evaluation metrics not meaningful."
        return metrics
    try:
        if len(scaled_data) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(scaled_data), 5000, replace=False)
            sample, sample_labels = scaled_data[idx], labels[idx]
        else:
            sample, sample_labels = scaled_data, labels

        metrics["silhouette_score"] = round(
            float(silhouette_score(sample, sample_labels)), 3
        )
        metrics["davies_bouldin_score"] = round(
            float(davies_bouldin_score(sample, sample_labels)), 3
        )
        metrics["calinski_harabasz_score"] = round(
            float(calinski_harabasz_score(sample, sample_labels)), 2
        )
        metrics["quality"] = (
            "Strong separation"
            if metrics["silhouette_score"] > 0.5
            else "Moderate overlap"
        )
    except Exception as e:
        metrics["evaluation_error"] = str(e)
    return metrics


def segment_customers(
    df: pd.DataFrame, method: str = "rules", n_clusters: int = 3
) -> tuple[pd.DataFrame, dict]:
    """
    Segments customers into Priority, Regular, and Dormant groups using one of:
    'rules', 'kmeans', 'hierarchical', or 'dbscan'.
    Returns a tuple of (segmented_dataframe, evaluation_metrics_dict).
    """
    df_segmented = df.copy()
    eval_metrics = {"method": method}
    logger.info(
        f"Running segmentation | method={method} n_clusters={n_clusters} "
        f"rows={len(df_segmented)}"
    )

    if method == "rules":

        def assign_segment(row):
            bal = row.get("avg_balance", row.get("current_balance", 0))
            freq = row.get("transaction_frequency", 0)
            if bal > PRIORITY_BALANCE_THRESHOLD or (
                bal > PRIORITY_BALANCE_WITH_FREQ and freq > PRIORITY_FREQ_THRESHOLD
            ):
                return "Priority"
            elif freq <= 1 or bal < 1000:
                return "Dormant"
            else:
                return "Regular"

        df_segmented["Segment"] = df_segmented.apply(assign_segment, axis=1)
        eval_metrics["info"] = "Rule-based logic applied cleanly."
        eval_metrics["rules_used"] = {
            "Priority": (
                f"avg_balance > {PRIORITY_BALANCE_THRESHOLD} OR "
                f"(avg_balance > {PRIORITY_BALANCE_WITH_FREQ} AND "
                f"transaction_frequency > {PRIORITY_FREQ_THRESHOLD})"
            ),
            "Dormant": "transaction_frequency <= 1 OR avg_balance < 1000",
            "Regular": "everything else",
        }
        eval_metrics["segment_distribution"] = (
            df_segmented["Segment"].value_counts().to_dict()
        )
        return df_segmented, eval_metrics

    num_cols = df_segmented.select_dtypes(include=["float64", "int64"]).columns
    if len(num_cols) < 2:
        eval_metrics[
            "evaluation_error"
        ] = "Not enough numeric columns for ML-based clustering."
        df_segmented["Segment"] = "Regular"
        return df_segmented, eval_metrics

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_segmented[num_cols])

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(scaled_data)
        df_segmented["Cluster"] = clusters
        label_map, ranking_cols = _rank_and_label_clusters(
            df_segmented, num_cols, scaled_data
        )
        df_segmented["Segment"] = df_segmented["Cluster"].map(label_map)
        eval_metrics["inertia"] = round(float(model.inertia_), 2)
        eval_metrics["ranking_columns_used"] = ranking_cols

    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = model.fit_predict(scaled_data)
        df_segmented["Cluster"] = clusters
        label_map, ranking_cols = _rank_and_label_clusters(
            df_segmented, num_cols, scaled_data
        )
        df_segmented["Segment"] = df_segmented["Cluster"].map(label_map)
        eval_metrics["ranking_columns_used"] = ranking_cols

    elif method == "dbscan":
        model = DBSCAN(eps=0.8, min_samples=5)
        clusters = model.fit_predict(scaled_data)
        df_segmented["Cluster"] = clusters
        # DBSCAN noise points (-1) are outliers/inactive — treat as Dormant.
        non_noise_mask = clusters != -1
        if non_noise_mask.sum() > 0 and len(set(clusters[non_noise_mask])) >= 1:
            label_map, ranking_cols = _rank_and_label_clusters(
                df_segmented[non_noise_mask],
                num_cols,
                scaled_data[non_noise_mask],
            )
        else:
            label_map, ranking_cols = {}, []
        label_map[-1] = "Dormant"
        df_segmented["Segment"] = (
            df_segmented["Cluster"].map(label_map).fillna("Regular")
        )
        eval_metrics["ranking_columns_used"] = ranking_cols
        eval_metrics["noise_points"] = int((clusters == -1).sum())

    else:
        raise ValueError(f"Unknown segmentation method: {method}")

    df_segmented.drop(columns=["Cluster"], inplace=True)
    eval_metrics["segment_distribution"] = (
        df_segmented["Segment"].value_counts().to_dict()
    )
    eval_metrics.update(_cluster_eval_metrics(scaled_data, clusters))
    return df_segmented, eval_metrics


def compare_clustering_methods(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """
    Runs KMeans, Hierarchical, and DBSCAN back-to-back and returns their
    evaluation metrics side by side.
    """
    results = {}
    for method in ["kmeans", "hierarchical", "dbscan"]:
        try:
            _, metrics = segment_customers(df, method=method, n_clusters=n_clusters)
            results[method] = metrics
        except Exception as e:
            results[method] = {"error": str(e)}
    return results


def find_upgrade_candidates(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Identifies Regular-segment customers closest to crossing into the Priority
    tier, ranked by a proximity score (balance + frequency relative to the
    Priority thresholds).
    """
    if "Segment" not in df.columns:
        return pd.DataFrame()

    regular_df = df[df["Segment"] == "Regular"].copy()
    if regular_df.empty:
        return regular_df

    bal_col = "avg_balance" if "avg_balance" in regular_df.columns else "current_balance"
    freq_col = (
        "transaction_frequency"
        if "transaction_frequency" in regular_df.columns
        else None
    )

    if bal_col not in regular_df.columns:
        return regular_df.head(top_n)

    regular_df["balance_gap_to_priority"] = (
        PRIORITY_BALANCE_THRESHOLD - regular_df[bal_col]
    ).clip(lower=0)

    if freq_col:
        regular_df["proximity_score"] = (
            regular_df[bal_col] / PRIORITY_BALANCE_THRESHOLD
        ) + (regular_df[freq_col] / PRIORITY_FREQ_THRESHOLD)
    else:
        regular_df["proximity_score"] = (
            regular_df[bal_col] / PRIORITY_BALANCE_THRESHOLD
        )

    return regular_df.sort_values("proximity_score", ascending=False).head(top_n)


def identify_edge_cases(df: pd.DataFrame) -> dict:
    """
    Identifies anomalous customers who don't fit cleanly into their assigned
    segment — the "edge cases" the hackathon architecture brief explicitly
    names as a Segmentation Tool deliverable.

    Returns flagged customers grouped by anomaly type, plus a human-readable
    summary. These are customers a bank analyst would want to manually review
    before trusting the automated segmentation.
    """
    if df is None or "Segment" not in df.columns:
        return {"error": "Dataset not segmented yet."}

    edge_cases = {
        "dormant_with_high_balance": [],
        "priority_with_low_activity": [],
        "high_frequency_low_balance": [],
        "segment_boundary_cases": [],
        "summary": "",
    }

    bal_col = "avg_balance" if "avg_balance" in df.columns else "current_balance"
    freq_col = (
        "transaction_frequency" if "transaction_frequency" in df.columns else None
    )
    id_col = next(
        (c for c in df.columns if "cust" in c.lower() or "id" in c.lower()),
        df.columns[0],
    )

    # 1. Dormant customers with surprisingly high balances (>₹20k)
    dormant = df[df["Segment"] == "Dormant"]
    high_bal_dormant = dormant[dormant[bal_col] > 20000]
    if not high_bal_dormant.empty:
        edge_cases["dormant_with_high_balance"] = (
            high_bal_dormant[[id_col, bal_col, freq_col]]
            .head(10)
            .round(2)
            .to_dict(orient="records")
        )

    # 2. Priority customers with suspiciously low activity (≤2 transactions)
    priority = df[df["Segment"] == "Priority"]
    if freq_col and freq_col in df.columns:
        inactive_priority = priority[priority[freq_col] <= 2]
        if not inactive_priority.empty:
            edge_cases["priority_with_low_activity"] = (
                inactive_priority[[id_col, bal_col, freq_col]]
                .head(10)
                .round(2)
                .to_dict(orient="records")
            )

        # 3. High-frequency (>20 txns) Regular customers with very low balance (<₹5k)
        regular = df[df["Segment"] == "Regular"]
        high_freq_reg = regular[
            (regular[freq_col] > 20) & (regular[bal_col] < 5000)
        ]
        if not high_freq_reg.empty:
            edge_cases["high_frequency_low_balance"] = (
                high_freq_reg[[id_col, bal_col, freq_col]]
                .head(10)
                .round(2)
                .to_dict(orient="records")
            )

    # 4. Customers straddling segment boundaries (proximity_score near 1.0)
    if "proximity_score" in df.columns:
        boundary = df[
            (df["proximity_score"] > 0.9) & (df["proximity_score"] < 1.1)
        ]
        if not boundary.empty:
            edge_cases["segment_boundary_cases"] = (
                boundary[[id_col, "Segment", bal_col, "proximity_score"]]
                .head(10)
                .round(3)
                .to_dict(orient="records")
            )

    # Build summary
    counts = {
        k: len(v)
        for k, v in edge_cases.items()
        if k != "summary" and isinstance(v, list)
    }
    total = sum(counts.values())

    if total == 0:
        edge_cases["summary"] = (
            "✅ No edge cases detected — all customers fit cleanly "
            "into their assigned segments."
        )
    else:
        parts = [f"Found **{total} edge cases** requiring manual review:"]
        if counts.get("dormant_with_high_balance", 0) > 0:
            parts.append(
                f"- {counts['dormant_with_high_balance']} Dormant customers with "
                f"unexpectedly high balances (>₹20k) — possible classification error "
                f"or accounts awaiting reactivation."
            )
        if counts.get("priority_with_low_activity", 0) > 0:
            parts.append(
                f"- {counts['priority_with_low_activity']} Priority customers with "
                f"≤2 transactions — may be dormant wealth accounts or data anomalies."
            )
        if counts.get("high_frequency_low_balance", 0) > 0:
            parts.append(
                f"- {counts['high_frequency_low_balance']} high-frequency Regular "
                f"customers who don't qualify as Priority — consider manual upgrade "
                f"review."
            )
        if counts.get("segment_boundary_cases", 0) > 0:
            parts.append(
                f"- {counts['segment_boundary_cases']} customers straddling segment "
                f"boundaries (proximity_score 0.9–1.1) — small changes in behavior "
                f"could shift their tier."
            )
        edge_cases["summary"] = "\n".join(parts)

    return edge_cases