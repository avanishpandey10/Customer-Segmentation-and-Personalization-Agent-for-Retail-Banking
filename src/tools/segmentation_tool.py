import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Columns that actually define "priority-ness" per the hackathon brief
# (balance being maintained + frequency of transactions), used to label
# KMeans clusters instead of trusting arbitrary column order.
RANKING_COLS_PRIORITY = ["avg_balance", "current_balance", "transaction_frequency"]

PRIORITY_BALANCE_THRESHOLD = 50000
PRIORITY_BALANCE_WITH_FREQ = 20000
PRIORITY_FREQ_THRESHOLD = 10


def segment_customers(df: pd.DataFrame, method: str = "rules", n_clusters: int = 3) -> tuple[pd.DataFrame, dict]:
    """
    Segments customers into Priority, Regular, and Dormant groups.
    Returns a tuple of (segmented_dataframe, evaluation_metrics_dict).
    """
    df_segmented = df.copy()
    eval_metrics = {"method": method}

    if method == "rules":
        def assign_segment(row):
            bal = row.get("avg_balance", row.get("current_balance", 0))
            freq = row.get("transaction_frequency", 0)

            if bal > PRIORITY_BALANCE_THRESHOLD or (bal > PRIORITY_BALANCE_WITH_FREQ and freq > PRIORITY_FREQ_THRESHOLD):
                return "Priority"
            elif freq <= 1 or bal < 1000:
                return "Dormant"
            else:
                return "Regular"

        df_segmented["Segment"] = df_segmented.apply(assign_segment, axis=1)
        eval_metrics["info"] = "Rule-based logic applied cleanly."
        eval_metrics["rules_used"] = {
            "Priority": f"avg_balance > {PRIORITY_BALANCE_THRESHOLD} OR (avg_balance > {PRIORITY_BALANCE_WITH_FREQ} AND transaction_frequency > {PRIORITY_FREQ_THRESHOLD})",
            "Dormant": "transaction_frequency <= 1 OR avg_balance < 1000",
            "Regular": "everything else",
        }
        # Even rule-based segmentation should report a basic "evaluation" —
        # required by the brief's "Model evaluation" functional requirement.
        eval_metrics["segment_distribution"] = df_segmented["Segment"].value_counts().to_dict()

    elif method == "kmeans":
        num_cols = df_segmented.select_dtypes(include=["float64", "int64"]).columns
        if len(num_cols) >= 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_segmented[num_cols])

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            df_segmented["Cluster"] = clusters

            # --- FIX: rank clusters using balance + frequency signal, not
            # an arbitrary "first numeric column". Falls back gracefully if
            # none of the expected columns exist. ---
            ranking_cols = [c for c in RANKING_COLS_PRIORITY if c in num_cols]
            if not ranking_cols:
                ranking_cols = [num_cols[0]]
            ranking_idxs = [list(num_cols).index(c) for c in ranking_cols]

            cluster_scores = (
                pd.Series(scaled_data[:, ranking_idxs].mean(axis=1), index=df_segmented.index)
                .groupby(df_segmented["Cluster"])
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

            df_segmented["Segment"] = df_segmented["Cluster"].map(label_map)
            df_segmented.drop(columns=["Cluster"], inplace=True)
            eval_metrics["ranking_columns_used"] = ranking_cols
            eval_metrics["segment_distribution"] = df_segmented["Segment"].value_counts().to_dict()

            # Model Evaluation Metrics
            try:
                # Subsample for fast evaluation calculation during live demos
                if len(scaled_data) > 5000:
                    rng = np.random.default_rng(42)
                    sample_idx = rng.choice(len(scaled_data), 5000, replace=False)
                    eval_sample = scaled_data[sample_idx]
                    sample_clusters = clusters[sample_idx]
                else:
                    eval_sample = scaled_data
                    sample_clusters = clusters

                sil_score = silhouette_score(eval_sample, sample_clusters)
                eval_metrics["silhouette_score"] = round(float(sil_score), 3)
                eval_metrics["inertia"] = round(float(kmeans.inertia_), 2)
                eval_metrics["quality"] = "Strong separation" if sil_score > 0.5 else "Moderate overlap"
            except Exception as e:
                eval_metrics["evaluation_error"] = str(e)

    return df_segmented, eval_metrics


def find_upgrade_candidates(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Identifies Regular-segment customers closest to crossing into the Priority
    tier, ranked by a proximity score (balance + frequency relative to the
    Priority thresholds). Answers: "Which regular customers can be converted
    to priority customers?"
    """
    if "Segment" not in df.columns:
        return pd.DataFrame()

    regular_df = df[df["Segment"] == "Regular"].copy()
    if regular_df.empty:
        return regular_df

    bal_col = "avg_balance" if "avg_balance" in regular_df.columns else "current_balance"
    freq_col = "transaction_frequency" if "transaction_frequency" in regular_df.columns else None

    if bal_col not in regular_df.columns:
        return regular_df.head(top_n)

    regular_df["balance_gap_to_priority"] = (PRIORITY_BALANCE_THRESHOLD - regular_df[bal_col]).clip(lower=0)

    if freq_col:
        regular_df["proximity_score"] = (
            regular_df[bal_col] / PRIORITY_BALANCE_THRESHOLD
        ) + (regular_df[freq_col] / PRIORITY_FREQ_THRESHOLD)
    else:
        regular_df["proximity_score"] = regular_df[bal_col] / PRIORITY_BALANCE_THRESHOLD

    return regular_df.sort_values("proximity_score", ascending=False).head(top_n)