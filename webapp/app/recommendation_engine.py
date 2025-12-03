"""
Recommendation Engine

Uses:
- Trained KMeans cluster model
- Clustered jobs parquet (with embeddings)
- SentenceTransformer embedder

Provides:
- recommend(cv_text, top_n) → top matching jobs
"""

from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:
    def __init__(
        self,
        models_dir: str = "models/trained",
        embedding_model_dir: str = "models/embeddings/all-mpnet-base-v2",
    ) -> None:
        """
        Initialize engine by loading:
        - the latest clustering model from models/trained/
        - the latest clustered job dataset from models/trained/
        - the SentenceTransformer embedding model

        This is aligned with:
          src/workers/model_training/train_cluster_model.py
        which saves:
          - job_cluster_model_<timestamp>.pkl
          - job_clusters_<timestamp>.parquet
        """
        models_dir_path = Path(models_dir)
        if not models_dir_path.exists():
            raise FileNotFoundError(
                f"Models directory not found: {models_dir_path}. "
                "Make sure you ran the training script."
            )

        # 1. Find latest model and clustered parquet
        model_path = self._find_latest_file(models_dir_path, "job_cluster_model_*.pkl")
        clusters_path = self._find_latest_file(models_dir_path, "job_clusters_*.parquet")

        print("\n🔧 Initializing RecommendationEngine...")
        print(f"📄 Loading model: {model_path}")
        self.cluster_model = joblib.load(model_path)

        print(f"📄 Loading clustered jobs: {clusters_path}")
        self.jobs_df = pd.read_parquet(clusters_path)

        # 2. Decide which columns to use for embeddings and cluster id
        self.embedding_col = self._detect_embedding_column(self.jobs_df)
        self.cluster_col = self._detect_cluster_column(self.jobs_df)

        print(f"   → Using embedding column: '{self.embedding_col}'")
        print(f"   → Using cluster column  : '{self.cluster_col}'")

        # 3. Load encoder model
        embedder_path = Path(embedding_model_dir)
        print(f"📄 Loading SentenceTransformer from: {embedder_path}")
        self.embedder = SentenceTransformer(str(embedder_path))

        print(f"✔ RecommendationEngine ready with {len(self.jobs_df)} jobs.")

    # ------------------------------------------------------------------ #
    # File & column detection helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_latest_file(directory: Path, pattern: str) -> Path:
        """Return the most recent file matching pattern in a directory."""
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {directory}.\n"
                "Run src/workers/model_training/train_cluster_model.py first."
            )
        return matches[-1]  # latest by name (timestamp in filename)

    @staticmethod
    def _detect_embedding_column(df: pd.DataFrame) -> str:
        """
        Detect which column holds the job embeddings.

        Priority:
          1) 'embedding_full'
          2) 'embedding'
          3) any object column with list/array-like values
        """
        if "embedding_full" in df.columns:
            return "embedding_full"
        if "embedding" in df.columns:
            return "embedding"

        # fallback: any column that stores list/array-like embeddings
        for col in df.columns:
            if df[col].dtype == "object":
                series = df[col].dropna()
                if not len(series):
                    continue
                first_val = series.iloc[0]
                if isinstance(first_val, (list, np.ndarray)) and len(first_val) > 10:
                    return col

        raise RuntimeError(
            "Could not detect an embedding column. "
            "Expected something like 'embedding_full' or 'embedding'."
        )

    @staticmethod
    def _detect_cluster_column(df: pd.DataFrame) -> str:
        """
        Detect which column holds the cluster labels.

        Priority:
          1) 'cluster_id'
          2) 'cluster'
        """
        if "cluster_id" in df.columns:
            return "cluster_id"
        if "cluster" in df.columns:
            return "cluster"

        raise RuntimeError(
            "Clustered jobs file is missing 'cluster_id' or 'cluster' column."
        )

    # ------------------------------------------------------------------ #
    # Recommendation API
    # ------------------------------------------------------------------ #

    def recommend(self, cv_text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend jobs for given free text (CV or keywords).

        Steps:
        1. Encode text → embedding
        2. Predict cluster using trained model
        3. Filter jobs to that cluster
        4. Compute cosine similarity within that cluster
        5. Return top_n jobs sorted by score
        """
        if not cv_text or not cv_text.strip():
            return []

        # 1) Embed text
        cv_embedding = self.embedder.encode(cv_text)
        cv_embedding = np.array(cv_embedding, dtype="float32").reshape(1, -1)

        # 2) Predict cluster
        cluster = self.cluster_model.predict(cv_embedding)[0]
        print(f"\n📌 Query assigned to cluster: {cluster}")

        # 3) Filter jobs in this cluster
        cluster_jobs = self.jobs_df[self.jobs_df[self.cluster_col] == cluster].copy()
        print(f"📦 Jobs in this cluster: {len(cluster_jobs)}")

        if cluster_jobs.empty:
            return []

        # 4) Compute cosine similarity
        job_embeddings = np.vstack(
            cluster_jobs[self.embedding_col].apply(lambda x: np.array(x, dtype="float32")).values
        )
        scores = cosine_similarity(cv_embedding, job_embeddings)[0]
        cluster_jobs["score"] = scores

        # 5) Sort & select top_n
        cluster_jobs.sort_values("score", ascending=False, inplace=True)

        if top_n <= 0:
            top_n = 10
        top_n = min(top_n, len(cluster_jobs))

        top_jobs = cluster_jobs.head(top_n)

        # Convert rows to simple dicts
        results: List[Dict[str, Any]] = []
        for _, row in top_jobs.iterrows():
            results.append(
                {
                    "id": row.get("job_id") or row.get("id"),
                    "title": row.get("title") or row.get("position") or row.get("job_title"),
                    "company": row.get("company"),
                    "location": row.get("location_display") or row.get("location"),
                    "country": row.get("location_country") or row.get("country"),
                    "category": row.get("category_label") or row.get("category"),
                    "description": row.get("description"),
                    "score": float(row["score"]),
                    "cluster": int(row[self.cluster_col]),
                    "url": row.get("redirect_url") or row.get("url"),
                    "source": row.get("source"),
                    "salary_min": row.get("salary_min"),
                    "salary_max": row.get("salary_max"),
                    "created_date": row.get("created_date"),
                }
            )

        print(f"✅ Returning {len(results)} recommended jobs.")
        return results
