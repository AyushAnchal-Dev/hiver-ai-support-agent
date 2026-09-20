"""
Baseline 2: TF-IDF Centroid Classifier implemented with pure NumPy and Python standard library.
"""
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
import numpy as np

def tokenize(text: str) -> List[str]:
    """Basic alphanumeric tokenization and lowercasing."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

class TfidfClassifier:
    """TF-IDF vector space classifier with class centroid matching."""

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.class_centroids: Dict[str, np.ndarray] = {}
        self.classes: List[str] = []

    def fit(self, texts: List[str], labels: List[str]):
        """Build vocabulary, compute IDF, and calculate class centroids."""
        self.classes = sorted(list(set(labels)))
        n_docs = len(texts)

        # 1. Document Frequency
        df_counts = Counter()
        tokenized_docs = [tokenize(t) for t in texts]
        for tokens in tokenized_docs:
            for term in set(tokens):
                df_counts[term] += 1

        # 2. Vocabulary
        filtered_terms = [t for t, cnt in df_counts.items() if cnt >= self.min_df]
        self.vocab = {term: idx for idx, term in enumerate(sorted(filtered_terms))}
        vocab_size = len(self.vocab)

        if vocab_size == 0:
            return

        # 3. Compute IDF
        idf_vals = np.zeros(vocab_size)
        for term, idx in self.vocab.items():
            df = df_counts[term]
            idf_vals[idx] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self.idf = idf_vals

        # 4. Transform documents to TF-IDF vectors
        doc_vectors = np.zeros((n_docs, vocab_size))
        for i, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf_counts = Counter(tokens)
            total_tokens = len(tokens)
            vec = np.zeros(vocab_size)
            for t, cnt in tf_counts.items():
                if t in self.vocab:
                    vec[self.vocab[t]] = (cnt / total_tokens) * self.idf[self.vocab[t]]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            doc_vectors[i] = vec

        # 5. Compute class centroids
        for cls in self.classes:
            indices = [i for i, lbl in enumerate(labels) if lbl == cls]
            if indices:
                centroid = np.mean(doc_vectors[indices], axis=0)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 0:
                    centroid /= c_norm
                self.class_centroids[cls] = centroid

    def transform(self, text: str) -> np.ndarray:
        """Transforms a single text string into an L2-normalized TF-IDF vector."""
        if not self.vocab or len(self.vocab) == 0:
            return np.array([])
        tokens = tokenize(text)
        vec = np.zeros(len(self.vocab))
        if not tokens:
            return vec
        tf_counts = Counter(tokens)
        total_tokens = len(tokens)
        for t, cnt in tf_counts.items():
            if t in self.vocab:
                vec[self.vocab[t]] = (cnt / total_tokens) * self.idf[self.vocab[t]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent via cosine similarity to class centroids."""
        if not self.class_centroids:
            return {
                "predicted_intent": "UNKNOWN_OTHER",
                "confidence": 0.0,
                "similarities": {}
            }

        vec = self.transform(text)
        if np.linalg.norm(vec) == 0:
            return {
                "predicted_intent": "UNKNOWN_OTHER",
                "confidence": 0.20,
                "similarities": {}
            }

        similarities: Dict[str, float] = {}
        for cls, centroid in self.class_centroids.items():
            sim = float(np.dot(vec, centroid))
            similarities[cls] = round(max(0.0, sim), 4)

        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_sim = sorted_sims[0]

        # Softmax over top similarities for calibrated confidence
        top_sims_vals = np.array([s for _, s in sorted_sims[:5]]) * 5.0  # temperature scaling
        exp_sims = np.exp(top_sims_vals - np.max(top_sims_vals))
        probs = exp_sims / np.sum(exp_sims)
        confidence = round(float(probs[0]), 2)

        return {
            "predicted_intent": top_intent,
            "confidence": min(0.95, max(0.25, confidence if top_sim > 0.1 else 0.35)),
            "similarity_score": round(top_sim, 4),
            "top_candidates": sorted_sims[:3]
        }
