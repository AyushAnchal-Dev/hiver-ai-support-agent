"""
BM25 Lexical Search Implementation.
Provides fast inverted-index keyword retrieval over conversation text.
"""
import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def tokenize(text: str) -> List[str]:
    """Tokenizes alphanumeric terms, lowercased, length >= 2."""
    if not text:
        return []
    # Strip URL placeholders and handles
    cleaned = re.sub(r"@\w+|<LINK:[^>]+>", " ", text)
    return re.findall(r"\b[a-zA-Z0-9]{2,}\b", cleaned.lower())

class BM25LexicalSearch:
    """Pure-Python / NumPy BM25 Inverted Index with corpus caching."""

    _shared_corpus_cache: Dict[str, Dict[str, Any]] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    index_loaded_from_disk: bool = False

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: np.ndarray = np.array([])
        self.avg_doc_len: float = 0.0
        self.num_docs: int = 0
        self.doc_ids: List[str] = []
        self.idf: Dict[str, float] = {}
        # Inverted index: term -> list of (doc_idx, tf)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    @classmethod
    def reset_cache(cls):
        """Resets the corpus cache and counters."""
        cls._shared_corpus_cache.clear()
        cls.cache_hits = 0
        cls.cache_misses = 0
        cls.index_loaded_from_disk = False

    def fit(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """Builds inverted index and calculates IDF statistics (cached if corpus repeated)."""
        self.num_docs = len(documents)
        self.doc_ids = doc_ids if doc_ids is not None else [str(i) for i in range(self.num_docs)]

        # Check shared cache
        first_id = self.doc_ids[0] if self.doc_ids else ""
        cache_key = f"{self.num_docs}_{first_id}_{self.k1}_{self.b}"
        if cache_key in self._shared_corpus_cache:
            BM25LexicalSearch.cache_hits += 1
            cached = self._shared_corpus_cache[cache_key]
            self.doc_len = cached["doc_len"]
            self.avg_doc_len = cached["avg_doc_len"]
            self.idf = cached["idf"]
            self.inverted_index = cached["inverted_index"]
            self._doc_len_factor = cached.get("_doc_len_factor")
            self.inverted_arrays = cached.get("inverted_arrays", {})
            self._query_cache: Dict[str, List[Tuple[str, float, int]]] = {}
            return

        BM25LexicalSearch.cache_misses += 1
        
        doc_lengths = np.zeros(self.num_docs, dtype=np.int32)
        doc_freqs = Counter()

        for idx, doc in enumerate(documents):
            tokens = tokenize(doc)
            doc_lengths[idx] = len(tokens)
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term].append((idx, count))
                doc_freqs[term] += 1

        self.doc_len = doc_lengths
        self.avg_doc_len = float(np.mean(doc_lengths)) if self.num_docs > 0 else 1.0

        # Precompute Lucene/BM25 IDF
        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

        # Vectorized precomputations for sub-millisecond retrieval
        self._doc_len_factor = (self.k1 * (1.0 - self.b + self.b * (self.doc_len / self.avg_doc_len))).astype(np.float32)
        self.inverted_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for term, postings in self.inverted_index.items():
            self.inverted_arrays[term] = (
                np.array([p[0] for p in postings], dtype=np.int32),
                np.array([p[1] for p in postings], dtype=np.float32)
            )
        self._query_cache: Dict[str, List[Tuple[str, float, int]]] = {}

        # Store in shared cache
        self._shared_corpus_cache[cache_key] = {
            "doc_len": self.doc_len,
            "avg_doc_len": self.avg_doc_len,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
            "_doc_len_factor": self._doc_len_factor,
            "inverted_arrays": self.inverted_arrays
        }

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float, int]]:
        """
        Executes vectorized BM25 search over query terms.
        Returns:
            List of (doc_id, bm25_score, doc_index) sorted in descending score order.
        """
        if not query or self.num_docs == 0:
            return []

        # Check in-memory query cache for instant warm queries
        if hasattr(self, "_query_cache") and query in self._query_cache:
            return self._query_cache[query]

        tokens = tokenize(query)
        if not tokens:
            return []

        query_terms = set(tokens)
        scores = np.zeros(self.num_docs, dtype=np.float32)
        has_matches = False

        for term in query_terms:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            if idf_val <= 0.0:
                continue

            if hasattr(self, "inverted_arrays") and term in self.inverted_arrays:
                doc_indices, tf_vals = self.inverted_arrays[term]
                denom = tf_vals + self._doc_len_factor[doc_indices]
                term_scores = idf_val * (tf_vals * (self.k1 + 1.0) / denom)
                np.add.at(scores, doc_indices, term_scores)
                has_matches = True
            elif term in self.inverted_index:
                postings = self.inverted_index[term]
                for doc_idx, tf in postings:
                    d_len = self.doc_len[doc_idx]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avg_doc_len))
                    scores[doc_idx] += idf_val * (numerator / denominator)
                has_matches = True

        if not has_matches:
            return []

        nonzero_indices = np.flatnonzero(scores)
        if len(nonzero_indices) == 0:
            return []

        if len(nonzero_indices) > top_k:
            part_idx = np.argpartition(-scores[nonzero_indices], top_k)[:top_k]
            top_subset = nonzero_indices[part_idx]
            sorted_subset = top_subset[np.argsort(-scores[top_subset])]
        else:
            sorted_subset = nonzero_indices[np.argsort(-scores[nonzero_indices])]

        results = [(self.doc_ids[idx], round(float(scores[idx]), 4), int(idx)) for idx in sorted_subset]
        
        if hasattr(self, "_query_cache"):
            if len(self._query_cache) >= 1024:
                self._query_cache.clear()
            self._query_cache[query] = results

        return results
