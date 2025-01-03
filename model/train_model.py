# file: model/train_model.py

"""
This script trains a RandomForestClassifier to predict whether a candidate is
disqualified (Yes/No) and if fits in a position based on textual data from the candidate's resume_text.

Steps:
1) Load and preprocess data from 'candidates.csv'.
2) Combine textual fields if needed to create 'resume_text'.
3) Clean 'Disqualified' from Yes/No to 1/0 and add a feature 'feat_skills_empty' (1 if Skills is empty, 0 otherwise).
4) Train a RandomForestClassifier pipeline with TF-IDF for the text.
5) Save pipeline to 'pipeline.pkl'.
"""

import os
import pickle

import pandas as pd

import re
import nltk

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# Make sure to download NLTK data if not done:
#  python -m nltk.downloader stopwords
#  python -m nltk.downloader punkt

from model.transformers import TextCleanerTransformer


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Loads candidate data from CSV.
    - Checks for 'Disqualified' column (Yes/No) -> (1/0).
    - Merges textual fields (Summary, Experiences, Skills, Keywords)
        into 'resume_text'.
    """
    df = pd.read_csv(csv_path).fillna("")

    # Convert 'Disqualified' = Yes/No -> 1/0 if present
    if "Disqualified" not in df.columns:
        raise ValueError("CSV must have 'Disqualified' column (Yes/No).")

    def map_disqualified(val):
        val_str = str(val).strip().lower()
        if val_str == "yes":
            return 1
        elif val_str == "no":
            return 0
        return 0  # default 0 if unexpected

    def check_skills_empty(skills_str: str) -> int:
        if isinstance(skills_str, str) and skills_str.strip() == "":
            return 0
        else:
            return 1

    df["Disqualified"] = df["Disqualified"].apply(map_disqualified)
    df["feat_skills_empty"] = df["Skills"].apply(check_skills_empty)

    # Merge textual fields -> 'resume_text'
    def merge_text(row, qna_pairs):
        summary = row.get("Summary", "")
        experiences = row.get("Experiences", "")
        skills = row.get("Skills", "")
        keywords = row.get("Keywords", "")

        # Merge all text fields
        merged = f"{summary} {experiences} {skills} {keywords}"

        # Every Par (QuestionN, AnswerN)
        for num, (qcol, acol) in qna_pairs.items():
            question_str = row.get(qcol, "")
            answer_str = row.get(acol, "")
            # Add to merged text
            merged += f" Q{num}: {question_str} A{num}: {answer_str}"

        return merged.strip()

    question_cols = []
    answer_cols = []
    # Create a dict: {number -> (col_question, col_answer)}
    qna_pairs = {}

    # Ex: "Question 1" -> group(1) = "1"
    question_pattern = re.compile(r"question\s*(\d+)", re.IGNORECASE)
    answer_pattern = re.compile(r"answer\s*(\d+)", re.IGNORECASE)

    for qcol in question_cols:
        match_q = question_pattern.search(qcol.lower())
        if match_q:
            qnum = match_q.group(1)  # ex. "1"
            for acol in answer_cols:
                match_a = answer_pattern.search(acol.lower())
                if match_a and match_a.group(1) == qnum:
                    # Emparejamos
                    qna_pairs[qnum] = (qcol, acol)
                    break

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower.startswith("question"):
            question_cols.append(col)
        elif col_lower.startswith("answer"):
            answer_cols.append(col)

    if "resume_text" not in df.columns:
        df["resume_text"] = df.apply(lambda row: merge_text(row, qna_pairs), axis=1)

    return df


def build_pipeline() -> Pipeline:
    """
    Builds a scikit-learn Pipeline:
      1) TextCleanerTransformer
      2) TfidfVectorizer
      3) RandomForestClassifier
    No numeric cols used here, but you can expand if needed.
    """
    text_pipeline = Pipeline(
        [
            ("cleaner", TextCleanerTransformer()),
            ("tfidf", TfidfVectorizer(max_features=2000)),
        ]
    )

    pipeline = Pipeline(
        [
            ("text_only", text_pipeline),
            (
                "rf_classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )

    return pipeline


def main():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")

    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "../data/candidates.csv")
    pipeline_path = os.path.join(current_dir, "pipeline.pkl")

    # 1) Load data
    df = load_and_prepare_data(csv_path)

    # 2) Separate X,y
    X = df["resume_text"] + df["feat_skills_empty"].astype(str)
    y = df["Disqualified"].values

    # 3) Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Build pipeline
    pipeline = build_pipeline()

    # 5) Fit pipeline
    pipeline.fit(X_train, y_train)

    # 6) Evaluate
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
    print(f"[DEBUG] CV accuracy (train set): {scores.mean():.3f}")

    val_preds = pipeline.predict(X_val)
    val_accuracy = (val_preds == y_val).mean()
    print(f"[DEBUG] Validation accuracy: {val_accuracy:.3f}")

    # 7) Save pipeline
    with open(pipeline_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Pipeline saved to: {pipeline_path}")


def charge_pipeline():
    """
    Helper to load the classification pipeline from 'pipeline.pkl'.
    """
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "pipeline.pkl")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


if __name__ == "__main__":
    main()
