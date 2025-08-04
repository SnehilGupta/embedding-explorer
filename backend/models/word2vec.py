from gensim.models import Word2Vec

class Word2VecEmbedder:

    # Initializes the Word2Vec embedder with hyperparameters and sets up empty model placeholder
    def __init__(self, vector_size=100, window=5, min_count=5, sg=0):
        self.vector_size = vector_size # Embedding vector dimensions
        self.window = window # Context window size
        self.min_count = min_count # Minimum frequency to include a word
        self.sg = sg # Training algorithm: 0=CBOW, 1=Skip-gram
        self.model = None # Placeholder for the trained model

    # Trains the Word2Vec model on the provided corpus, handling both string and pre-tokenized inputs
    def fit(self, corpus):
        """Train the Word2Vec model"""
        # Ensure corpus is tokenized
        if isinstance(corpus[0], str):
            # If corpus contains strings, tokenize them
            tokenized_corpus = [text.split() for text in corpus]
        else:
            # Assume already tokenized
            tokenized_corpus = corpus

        self.model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=4
        )

    # Retrieves the numerical vector representation for a specific word from the trained model
    def get_vector(self, word):
        """Get vector for a word"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        try:
            return self.model.wv[word]
        except KeyError:
            raise KeyError(f"Word '{word}' not in vocabulary")

    # Utility method to check if a word exists in the model's vocabulary before attempting operations
    def has_word(self, word):
        """Check if word exists in vocabulary"""
        if self.model is None:
            return False
        return word in self.model.wv.key_to_index

    # Returns the complete list of words that the model knows
    def get_vocabulary(self):
        """Get list of words in vocabulary"""
        if self.model is None:
            return []
        return list(self.model.wv.key_to_index.keys())

    # Finds the most semantically similar words to a query word using cosine similarity in the embedding space
    def get_similar_words(self, word, top_n=10):
        """Find similar words"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if not self.has_word(word):
            # Get vocabulary sample for debugging
            vocab_sample = self.get_vocabulary()[:20]
            raise KeyError(f"Word '{word}' not in vocabulary. Sample words: {vocab_sample}")

        try:
            similar_words = self.model.wv.most_similar(word, topn=top_n)
            return [(word, float(similarity)) for word, similarity in similar_words]
        except Exception as e:
            raise ValueError(f"Error finding similar words: {str(e)}")
