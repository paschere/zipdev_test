# file: tests/unit/test_train_model.py

import pytest
import pandas as pd


from model.train_model import (
    load_and_prepare_data,
    build_pipeline,
)


def test_load_and_prepare_data(tmp_path):
    # Create a temporary CSV
    csv_file = tmp_path / "candidates.csv"
    data = {
        "Disqualified": ["Yes", "No"],
        "Summary": ["Python dev", "Ruby dev"],
        "Experiences": ["Exp 1", "Exp 2"],
        "Skills": ["Django", ""],
        "Keywords": ["backend", "rails"],
    }
    df_test = pd.DataFrame(data)
    df_test.to_csv(csv_file, index=False)

    # Call load_and_prepare_data
    df_loaded = load_and_prepare_data(str(csv_file))
    assert "resume_text" in df_loaded.columns
    # Check that "Yes" => 1, "No" => 0
    assert df_loaded["Disqualified"].tolist() == [1, 0]

    # feat_skills_empty => if skills is empty => 0, else 1
    # Actually your code sets 0 if empty, 1 if not empty
    assert df_loaded["feat_skills_empty"].tolist() == [1, 0]


def test_load_and_prepare_data_raises_value_error(tmp_path):
    # If "Disqualified" col is missing
    csv_file = tmp_path / "candidates_missing.csv"
    df = pd.DataFrame({"Summary": ["Some summary"]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="CSV must have 'Disqualified' column"):
        load_and_prepare_data(str(csv_file))


def test_build_pipeline():
    pipeline = build_pipeline()
    assert pipeline is not None
    # Check the named steps
    steps = pipeline.named_steps
    assert "text_only" in steps
    assert "rf_classifier" in steps

    # Inside text_only we have a sub-pipeline with "cleaner" and "tfidf"
    text_subpipeline = steps["text_only"]
    assert text_subpipeline.named_steps["cleaner"] is not None
    assert text_subpipeline.named_steps["tfidf"] is not None
