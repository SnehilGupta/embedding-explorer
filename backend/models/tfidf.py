from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFEmbedder:
    def __init__(self, max_features=10000, min_df=1, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,  # Include words that appear in at least 1 document
            max_df=max_df,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )
        self.model = None

    def fit(self, corpus):
        """Train the TF-IDF model"""
        # Join tokens if corpus is tokenized
        if isinstance(corpus[0], list):
            corpus = [' '.join(tokens) for tokens in corpus]

        self.model = self.vectorizer.fit_transform(corpus)

    def get_vector(self, word):
        """Get TF-IDF vector for a word"""
        if not self.has_word(word):
            raise KeyError(f"Word '{word}' not in vocabulary")

        word_idx = self.vectorizer.vocabulary_[word]
        # Return the word's column from the TF-IDF matrix
        return self.model[:, word_idx].toarray().flatten()

    def has_word(self, word):
        """Check if word exists in vocabulary"""
        if self.model is None:
            return False
        return word.lower() in self.vectorizer.vocabulary_

    def get_vocabulary(self):
        """Get list of words in vocabulary"""
        if self.model is None:
            return []
        return list(self.vectorizer.vocabulary_.keys())

    def get_similar_words(self, word, top_n=10):
        """Find similar words using cosine similarity"""
        if not self.has_word(word):
            vocab_sample = list(self.vectorizer.vocabulary_.keys())[:20]
            raise KeyError(f"Word '{word}' not in vocabulary. Sample words: {vocab_sample}")

        word_vector = self.get_vector(word).reshape(1, -1)
        similarities = []

        for vocab_word in self.vectorizer.vocabulary_.keys():
            if vocab_word != word:
                vocab_vector = self.get_vector(vocab_word).reshape(1, -1)
                similarity = cosine_similarity(word_vector, vocab_vector)[0][0]
                similarities.append((vocab_word, float(similarity)))

        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
