# file: tests/integration/test_integration.py

import pytest
import pandas as pd
from unittest.mock import patch

from model import job_descriptions


@pytest.fixture
def mock_candidates():
    # Minimal DataFrame that job_descriptions might use
    data = {
        "resume_text": [
            "ruby on rails developer",
            "node js and docker dev",
            "marketing specialist no tech",
        ],
        "Name": ["A", "B", "C"],
    }
    return pd.DataFrame(data)


def test_filter_candidates_mock_pipeline(mock_candidates, monkeypatch):
    """
    We want to test filter_candidates() which uses pipeline.predict
    globally and job_descriptions.CANDIDATES.
    We'll monkeypatch job_descriptions.CANDIDATES with mock_candidates,
    and mock pipeline.predict to simulate disqualifications.
    """
    # Temporarily replace the global CANDIDATES in job_descriptions
    monkeypatch.setattr(job_descriptions, "CANDIDATES", mock_candidates)

    # Mock the pipeline
    with patch.object(job_descriptions.pipeline, "predict", return_value=[0, 0, 1]):
        # 3 candidates => the third is disqualified
        df_filtered = job_descriptions.filter_candidates()

        assert len(df_filtered) == 2
        assert df_filtered.iloc[0]["Name"] == "A"
        assert df_filtered.iloc[1]["Name"] == "B"


def test_compute_similarity_integration(mock_candidates):
    """
    Test compute_similarity with a known job_desc
    """
    job_desc = "Looking for node.js developer with Docker experience"
    results = job_descriptions.compute_similarity(mock_candidates, job_desc)
    # We expect "node js and docker dev" to rank first
    assert len(results) == 2
    # The highest similarity should be for candidate B
    assert results[0]["Name"] == "B"
