import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Union

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a file

    Args:
        file_path: Path to the dataset file

    Returns:
        Pandas DataFrame containing the dataset

    Raises:
        ValueError: If the file format is not supported
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess text by tokenizing, removing stopwords, and converting to lowercase

    Args:
        text: Text to preprocess

    Returns:
        List of preprocessed tokens
    """
    if not isinstance(text, str):
        return []

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    return tokens

def reduce_dimensions(vectors: np.ndarray, method: str = 'pca', n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """
    Reduce the dimensionality of vectors

    Args:
        vectors: Array of vectors to reduce
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components in the reduced space
        random_state: Random state for reproducibility

    Returns:
        Array of reduced vectors

    Raises:
        ValueError: If the method is not supported
    """
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=min(30, len(vectors) - 1))
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    return reducer.fit_transform(vectors)

def create_visualization_data(words: List[str], vectors: np.ndarray, highlight_word: Optional[str] = None) -> Dict:
    """
    Create data for visualization

    Args:
        words: List of words
        vectors: Reduced vectors for the words
        highlight_word: Word to highlight in the visualization

    Returns:
        Dictionary with visualization data
    """
    result = {
        "words": words,
        "x": vectors[:, 0].tolist(),
        "y": vectors[:, 1].tolist(),
        "highlight": [word == highlight_word for word in words] if highlight_word else [False] * len(words)
    }

    return result
