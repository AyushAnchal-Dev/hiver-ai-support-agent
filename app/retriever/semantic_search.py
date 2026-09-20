"""
Semantic Dense Vector Search Module.
Integrates BAAI/bge-small-en-v1.5 embeddings with FAISS vector store.
"""
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.retriever.vector_store import FaissVectorStore

class SemanticSearch:
    """Dense semantic retriever using BAAI/bge-small-en-v1.5 and FAISS."""

    # Shared singletons across all instances
    _shared_models: Dict[str, Any] = {}
    _shared_vector_stores: Dict[str, FaissVectorStore] = {}
    _embedding_cache: OrderedDict = OrderedDict()
    _max_cache_size: int = 2048
    cache_hits: int = 0
    cache_misses: int = 0

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        index_path: Optional[str] = None,
        index_type: str = "flat",
        doc_ids: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.index_type = index_type
        self.doc_ids = doc_ids or []
        self._model = None
        self.vector_store: Optional[FaissVectorStore] = None

        if index_path and os.path.exists(index_path):
            self.load_index(index_path, index_type)

    @classmethod
    def cache_stats(cls) -> Dict[str, int]:
        """Returns embedding cache statistics."""
        return {
            "cache_hits": cls.cache_hits,
            "cache_misses": cls.cache_misses,
            "cache_size": len(cls._embedding_cache),
            "max_cache_size": cls._max_cache_size
        }

    @classmethod
    def clear_cache(cls):
        """Clears the embedding cache and resets counters."""
        cls._embedding_cache.clear()
        cls.cache_hits = 0
        cls.cache_misses = 0

    @property
    def cache_size(self) -> int:
        return len(self._embedding_cache)

    @classmethod
    def preload_model(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Preloads and caches SentenceTransformer model in memory."""
        if model_name not in cls._shared_models:
            from sentence_transformers import SentenceTransformer
            cls._shared_models[model_name] = SentenceTransformer(model_name)
        return cls._shared_models[model_name]

    @classmethod
    def warm_cache(cls, queries: List[str], model_name: str = "BAAI/bge-small-en-v1.5"):
        """Precomputes and caches vector embeddings for frequent/startup queries."""
        model = cls.preload_model(model_name)
        for q in queries:
            if q and q not in cls._embedding_cache:
                cls._embedding_cache[q] = model.encode(q, convert_to_numpy=True, normalize_embeddings=True)

    @property
    def model(self):
        """Lazy loader for SentenceTransformer model with class-level singleton cache."""
        if self.model_name not in SemanticSearch._shared_models:
            SemanticSearch.preload_model(self.model_name)
        return SemanticSearch._shared_models[self.model_name]

    def load_index(self, index_path: str, index_type: str = "flat"):
        """Loads FAISS index from disk or retrieves resident in-memory store."""
        self.index_type = index_type
        cache_key = f"{index_path}_{index_type}"
        if cache_key in self._shared_vector_stores:
            self.vector_store = self._shared_vector_stores[cache_key]
        else:
            loaded_store = FaissVectorStore.load(index_path, index_type=index_type)
            self._shared_vector_stores[cache_key] = loaded_store
            self.vector_store = loaded_store

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes and L2-normalizes an incoming query string with LRU caching."""
        if query in self._embedding_cache:
            SemanticSearch.cache_hits += 1
            self._embedding_cache.move_to_end(query)
            return self._embedding_cache[query]

        SemanticSearch.cache_misses += 1
        vec = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        if len(self._embedding_cache) >= self._max_cache_size:
            self._embedding_cache.popitem(last=False)

        self._embedding_cache[query] = vec
        return vec


    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float, int]]:
        """
        Retrieves top_k nearest neighbors by cosine similarity.
        Returns:
            List of (doc_id, cosine_similarity, index)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_index first.")

        query_vec = self.encode_query(query)
        scores, indices = self.vector_store.search(query_vec, top_k=top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or (self.doc_ids and idx >= len(self.doc_ids)):
                continue
            doc_id = self.doc_ids[idx] if self.doc_ids else str(idx)
            results.append((doc_id, float(score), int(idx)))
        return results
