# file: tests/unit/test_transformers.py

import pytest
import pandas as pd
from model.transformers import TextCleanerTransformer


@pytest.fixture
def sample_texts():
    return pd.Series(
        [
            "Hello WORLD!!! This is a test.   ",
            "  Ruby on Rails 123 ??? ",
            "Node.js, Docker & React???",
            None,
        ]
    )


def test_text_cleaner_transform_basic(sample_texts):
    transformer = TextCleanerTransformer()
    # "fit" does nothing, but we call it for completeness
    transformer.fit(sample_texts)

    cleaned = transformer.transform(sample_texts)
    assert len(cleaned) == 4
    assert isinstance(cleaned[0], str)

    # Check we have lowercased and removed special symbols
    # 1) "Hello WORLD!!! This is a test." => "hello world test"
    assert "hello" in cleaned[0]
    assert "world" in cleaned[0]
    assert "test" in cleaned[0]
    assert "!!!" not in cleaned[0]
    # 2) "Ruby on Rails 123 ???"
    #    Might become "ruby rails 123" (stopwords like "on" might be removed)
    assert "ruby" in cleaned[1]
    assert "rails" in cleaned[1]
    # "???" should not remain
    assert "???" not in cleaned[1]


def test_text_cleaner_transform_none(sample_texts):
    transformer = TextCleanerTransformer()
    cleaned = transformer.transform(sample_texts)
    # The last item is None, check we handle it gracefully
    assert (
        cleaned[3] == "" or cleaned[3] == "none"
    ), f"Expected an empty string for None, got {cleaned[3]}"
