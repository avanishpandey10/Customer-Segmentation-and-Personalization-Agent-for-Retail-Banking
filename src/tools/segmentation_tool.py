import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
            
            if bal > 50000 or (bal > 20000 and freq > 10):
                return "Priority"
            elif freq <= 1 or bal < 1000:
                return "Dormant"
            else:
                return "Regular"
                
        df_segmented["Segment"] = df_segmented.apply(assign_segment, axis=1)
        eval_metrics["info"] = "Rule-based logic applied cleanly."
        
    elif method == "kmeans":
        num_cols = df_segmented.select_dtypes(include=["float64", "int64"]).columns
        if len(num_cols) >= 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_segmented[num_cols])
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            df_segmented["Cluster"] = clusters
            cluster_means = df_segmented.groupby("Cluster")[num_cols[0]].mean().sort_values()
            
            label_map = {
                cluster_means.index[0]: "Dormant",
                cluster_means.index[-1]: "Priority"
            }
            for idx in cluster_means.index:
                if idx not in label_map:
                    label_map[idx] = "Regular"
                    
            df_segmented["Segment"] = df_segmented["Cluster"].map(label_map)
            df_segmented.drop(columns=["Cluster"], inplace=True)
            
            # Model Evaluation Metrics
            try:
                # Subsample for fast evaluation calculation during live demos
                eval_sample = scaled_data[:5000] if len(scaled_data) > 5000 else scaled_data
                sample_clusters = clusters[:5000] if len(clusters) > 5000 else clusters
                
                sil_score = silhouette_score(eval_sample, sample_clusters)
                eval_metrics["silhouette_score"] = round(float(sil_score), 3)
                eval_metrics["inertia"] = round(float(kmeans.inertia_), 2)
                eval_metrics["quality"] = "Strong separation" if sil_score > 0.5 else "Moderate overlap"
            except Exception as e:
                eval_metrics["evaluation_error"] = str(e)
            
    return df_segmented, eval_metrics