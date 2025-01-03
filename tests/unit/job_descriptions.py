# file: tests/unit/test_job_descriptions.py

import pytest
import numpy as np

from job_descriptions import (
    validate_job_description,
    clean_text,
    normalize_scores,
)


def test_validate_job_description():
    # Empty or none-like
    assert validate_job_description("") is False
    # Over 3500 chars
    too_long_text = "x" * 3501
    assert validate_job_description(too_long_text) is False
    # Valid
    valid_text = "Need a Node.js developer with Docker experience"
    assert validate_job_description(valid_text) is True


def test_clean_text():
    raw = "Hello WORLD!!! Node.js dev ???"
    cleaned = clean_text(raw)
    # Should be lowercased, special characters stripped
    assert cleaned == "hello world   node js dev"


def test_normalize_scores():
    # Typical scenario
    sims = np.array([0.1, 0.5, 0.9])
    normalized = normalize_scores(sims)
    # 0.1 -> 0, 0.9 -> 100, 0.5 -> ~50
    assert len(normalized) == 3
    assert normalized[0] == 0
    assert pytest.approx(normalized[2], 0.1) == 100.0
    assert pytest.approx(normalized[1], 0.1) == 50.0

    # All sims the same
    sims_same = np.array([0.4, 0.4, 0.4])
    norm_same = normalize_scores(sims_same)
    # Should all become 0
    assert np.all(norm_same == 0)
