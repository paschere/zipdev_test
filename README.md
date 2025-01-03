# Candidate Screening & Scoring System

This repository contains a **Python** application for:

- **Data ingestion** and preprocessing of candidate information.
- **Classification** of candidates using a scikit-learn **RandomForest** pipeline.
- **NLP-based** text cleaning (with `TextCleanerTransformer`).
- **Job description** similarity scoring (TF-IDF + heuristic).
- **API usage** (optional) for filtering and ranking candidates.

## Table of Contents

1.  [Project Structure](#project-structure)
2.  [Requirements](#requirements)
3.  [Installation](#installation)
4.  [Usage](#usage)
5.  [Key Scripts](#key-scripts)
    - [`train_model.py`](#train_modelpy)
    - [`job_descriptions.py`](#job_descriptionspy)
    - [`transformers.py`](#transformerspy)
6.  [Testing](#testing)
7.  [Notes on TF-IDF Similarity vs. Model Classification](#notes-on-tf-idf-similarity-vs-model-classification)
8.  [Deployment Tips](#deployment-tips)

---

## Project Structure

Below is a sample layout of this repository:

```python-test/
├── data/
│   └── candidates.csv
├── frontend/
│   ├── App.jsx             # Frontend with React using Vite
├── model/
│   ├── __init__.py
│   ├── train_model.py
│   ├── transformers.py
|   |── job_descriptions.py
│   └── pipeline.pkl         # Generated after training (ignored if not yet created)
├── job_descriptions.py
├── tests/
│   ├── integration/
│   │   └── test_integration.py
│   └── unit/
│       ├── test_transformers.py
│       ├── test_job_descriptions.py
│       └── test_train_model.py
├── requirements.txt
├── README.md                # <-- You are here
└── ...
```

**Key files**:

- `model/train_model.py`: Orchestrates RandomForest training, loads data, etc.
- `model/transformers.py`: Defines `TextCleanerTransformer`, a custom scikit-learn transformer for text cleaning.
- `job_descriptions.py`: Contains logic for filtering candidates, computing similarity, cleaning job descriptions, etc.
- `tests/`: Directory containing unit and integration tests.

---

## Requirements

- **Python 3.9+** (recommended)
- **pip** or **conda** to install Python packages
- **NLTK** stopwords/punkt data (download with `nltk.download('stopwords')`, `nltk.download('punkt')`)

---

## Installation

1.  **Clone** the repository:

    ```
    git clone https://github.com/<yourusername>/python-test.git
     cd python-test
    ```

2.  **Create a virtual environment** (recommended):

    ```
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate`
    ```

3.  **Install** dependencies:

    ```
        pip install -r requirements.txt
    ```

4.  **Download** NLTK data (stopwords, punkt), if not already installed:

        ```
        python -m nltk.downloader stopwords
        python -m nltk.downloader punkt
        ```

---

## Usage

### 1) Train the Model

Run the `train_model.py` script to build and save the pipeline model (`pipeline.pkl`):

```
python -m model.train_model

```

**What it does**:

- Loads `candidates.csv` from `../data/`.
- Merges relevant text fields into `resume_text`.
- Adds a binary feature if `Skills` is empty or not.
- Trains a **RandomForest** pipeline that includes:
  - `TextCleanerTransformer`
  - `TfidfVectorizer`
  - RandomForestClassifier
- Saves the trained pipeline as `pipeline.pkl` for later usage.

### 2) Filter & Score Candidates

You can then run or import `job_descriptions.py` functions. For instance:

```python
# Example usage (pseudo-code)
from job_descriptions import filter_candidates, compute_similarity

# 1) Filter out disqualified candidates:

df_not_disqualified = filter_candidates()

# 2) Compute similarity ranking for a given job description:

job_desc = "Looking for Node.js developer with Docker experience"
top_candidates = compute_similarity(df_not_disqualified, job_desc)
for cand in top_candidates:
print(cand["Name"], cand["similarity_score"])
```

---

## Key Scripts

### `train_model.py`

- **Location**: `model/train_model.py`
- **Purpose**:
  - Loads `candidates.csv`.
  - Converts “Yes”/“No” in `Disqualified` to `1/0`.
  - Builds a pipeline with `TextCleanerTransformer` + TF-IDF + RandomForest.
  - Saves to `pipeline.pkl`.

Usage:

```
    python -m model.train_model

```

### `job_descriptions.py`

- **Purpose**:
  - Contains global `CANDIDATES` (from `candidates.csv`).
  - `filter_candidates()`: uses `pipeline.predict` to remove those predicted as disqualified.
  - `compute_similarity()`: uses TF-IDF on each candidate’s `resume_text` plus the new job description, calculates cosine similarity, normalizes scores, and returns top 30.
  - `validate_job_description()`: checks length constraints.
  - `clean_text()`: basic cleaning function.
  - `normalize_scores()`: scales similarity [0..1] => [0..100].

### `transformers.py`

- **Purpose**:
  - Defines `TextCleanerTransformer`, a custom scikit-learn `BaseEstimator` + `TransformerMixin`.
  - Steps:
    1.  Lowercase
    2.  Remove non-alphanumeric
    3.  Tokenize
    4.  Remove stopwords
    5.  Re-join
  - Can be used in any `Pipeline`.

---

## Testing

The project is using **`pytest`**. Ensure you installed it:

`pip install pytest pytest-cov`

### Run All Tests

From the project root (`python-test/`), run:

`python -m pytest --cov=. --cov-report=term-missing`

or `pytest --cov=. --cov-report=html` for an HTML coverage report.

**Tests** are divided into:

1.  **Unit tests** (`tests/unit/`):

    - `test_transformers.py`: checks `TextCleanerTransformer`.
    - `test_job_descriptions.py`: checks small functions (`validate_job_description`, `clean_text`, etc.).
    - `test_train_model.py`: checks `load_and_prepare_data`, `build_pipeline`, and possible CSV logic.

2.  **Integration tests** (`tests/integration/`):

    - `test_integration.py`: uses partial mocking or monkeypatching to test how multiple pieces fit together (e.g., `filter_candidates` calling `pipeline.predict`, then `compute_similarity`).

Example commands:

```
# from project root
pytest tests/unit/
pytest tests/integration/
```

---

## Notes on TF-IDF Similarity vs. Model Classification

- **Model classification** (`pipeline.predict`) sets “disqualified” vs. “not disqualified.”
- **TF-IDF similarity** (`compute_similarity`) ranks “not disqualified” candidates by textual similarity to the job description.
- Combined:
  1.  Filter out disqualified (`filter_candidates`).
  2.  Among the rest, do `compute_similarity`.
