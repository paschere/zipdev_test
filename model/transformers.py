import pandas as pd
import re
from typing import List
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.corpus import stopwords


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that cleans text by:
      1. Lowercasing
      2. Removing non-alphanumeric characters
      3. Tokenizing (via NLTK)
      4. Removing English stopwords
      5. Re-joining tokens into a single string

    This can be placed inside a Pipeline to automatically clean text columns.
    """

    def fit(self, X: pd.Series, y=None):
        """
        No fitting logic is required for text cleaning, so we simply return `self`.
        """
        return self

    def transform(self, X: pd.Series) -> List[str]:
        """
        Applies text cleaning transformations to each row of input data.

        Args:
            X (pd.Series): A pandas Series containing textual data.

        Returns:
            List[str]: A list of cleaned text strings (one per row).
        """
        cleaned_texts = []
        # Ensure NLTK data is downloaded
        eng_stopwords = set(stopwords.words("english"))

        for text in X:
            # Convert to string and lowercase
            text = str(text).lower()
            # Remove any non-alphanumeric characters
            text = re.sub(r"[^a-z0-9\s]", " ", text)
            # Tokenize
            tokens = word_tokenize(text)
            # Remove stopwords and empty tokens
            tokens = [t for t in tokens if t not in eng_stopwords and t.strip() != ""]
            # Re-join tokens into a single string
            cleaned_texts.append(" ".join(tokens))

        return cleaned_texts
