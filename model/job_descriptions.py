# Load necessary data and models
import os
import re
from typing import Any, Dict, List

import pandas as pd
from model.train_model import charge_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


pipeline = charge_pipeline()
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/candidates.csv")
CANDIDATES = pd.read_csv(DATA_PATH).fillna("")


def merge_candidate_text(row: pd.Series) -> str:
    """
    Combines relevant fields into a single text block for analysis.
    """
    summary = row.get("Summary", "")
    experiences = row.get("Experiences", "")
    skills = row.get("Skills", "")
    keywords = row.get("Keywords", "")
    return f"{summary} {experiences} {skills} {keywords}".strip()


def clean_text(text: str) -> str:
    """
    Cleans input text by removing non-alphanumeric characters and
    converting to lowercase.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()


def validate_job_description(job_desc: str) -> bool:
    """
    Validates the length of the job description.
    """
    return bool(job_desc) and len(job_desc) <= 3500


def init_candidate_data():
    """
    Load candidate data and process it for analysis.
    """

    if "resume_text" not in CANDIDATES.columns:
        CANDIDATES["resume_text"] = CANDIDATES.apply(merge_candidate_text, axis=1)
    # Process the candidate data
    CANDIDATES["resume_text"] = CANDIDATES["resume_text"].apply(clean_text)


def filter_candidates() -> pd.DataFrame:
    """
    Filters out candidates predicted as disqualified
    by the classification pipeline.
    """
    df_temp = CANDIDATES.copy()
    df_temp["disqualified_pred"] = pipeline.predict(df_temp["resume_text"])

    return df_temp[df_temp["disqualified_pred"] == 0].copy()


def compute_similarity(df: pd.DataFrame, job_desc: str) -> List[Dict[str, Any]]:
    """
    Computes TF-IDF similarity scores for candidates and
    returns the top 30.
    """
    job_desc_clean = clean_text(job_desc)
    corpus = df["resume_text"].tolist() + [job_desc_clean]

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(corpus)

    job_vec = X[-1]
    candidate_vecs = X[:-1]

    sims = cosine_similarity(candidate_vecs, job_vec).flatten()

    df["similarity_score"] = normalize_scores(sims)
    df_sorted = df[df["similarity_score"] > 0].sort_values(
        "similarity_score", ascending=False
    )

    return df_sorted.head(30).to_dict("records")


def normalize_scores(sims: pd.Series) -> pd.Series:
    """
    Normalizes similarity scores to a range of 0 to 100.
    """
    min_s, max_s = sims.min(), sims.max()
    return (100 * (sims - min_s) / (max_s - min_s)) if max_s != min_s else sims * 0
