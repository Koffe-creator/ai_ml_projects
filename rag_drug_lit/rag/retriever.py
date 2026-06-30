"""Finds the passages in our corpus that best match a question.

We use TF-IDF (a simple way to turn text into numbers based on word
importance) and cosine similarity to rank passages. No vector database,
nothing fancy - just scikit-learn.
"""

import os
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(text, words_per_chunk=120, overlap=30):
    """Break a long document into smaller overlapping pieces.

    Overlap means consecutive chunks share some words, so we don't lose
    a sentence that happens to fall on a chunk boundary.
    """
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        piece = words[start:start + words_per_chunk]
        chunks.append(" ".join(piece))
        start = start + words_per_chunk - overlap
    return chunks


class Retriever:
    def __init__(self, corpus_dir):
        # Load every .txt file and chop it into chunks.
        # We keep two parallel lists: the chunk text, and which file it came from.
        self.chunk_texts = []
        self.chunk_sources = []

        for path in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
            filename = os.path.basename(path)
            with open(path) as f:
                content = f.read()
            for chunk in split_into_chunks(content):
                self.chunk_texts.append(chunk)
                self.chunk_sources.append(filename)

        if not self.chunk_texts:
            raise ValueError("No .txt files found in " + corpus_dir)

        # Turn all chunks into TF-IDF vectors once, up front.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunk_texts)

    def search(self, question, k=3):
        """Return the top-k passages for a question.

        Each result is a dict with the source filename, the text, and the
        similarity score (higher = more relevant).
        """
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.chunk_vectors)[0]

        # Find the indices of the highest-scoring chunks.
        best = scores.argsort()[::-1][:k]

        results = []
        for i in best:
            results.append({
                "source": self.chunk_sources[i],
                "text": self.chunk_texts[i],
                "score": float(scores[i]),
            })
        return results
