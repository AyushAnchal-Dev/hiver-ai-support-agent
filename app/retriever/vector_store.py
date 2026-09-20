"""
FAISS Vector Store wrapper supporting IndexFlatIP and IndexHNSWFlat.
Provides serialization, deserialization, top-k search, and benchmarking stats.
"""
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

class FaissVectorStore:
    """Production vector store wrapper around FAISS indices."""

    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type.lower()
        self.index = None

    @classmethod
    def build_flat_index(cls, embeddings: np.ndarray) -> "FaissVectorStore":
        """Builds an exact IndexFlatIP for L2-normalized vectors."""
        if faiss is None:
            raise ImportError("faiss is not installed. Install faiss-cpu first.")
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
        dim = embeddings.shape[1]
        store = cls(dimension=dim, index_type="flat")
        store.index = faiss.IndexFlatIP(dim)
        emb_f32 = np.ascontiguousarray(embeddings.astype("float32"))
        store.index.add(emb_f32)
        return store

    @classmethod
    def build_hnsw_index(cls, embeddings: np.ndarray, m: int = 32, ef_construction: int = 64) -> "FaissVectorStore":
        """Builds an approximate IndexHNSWFlat graph index."""
        if faiss is None:
            raise ImportError("faiss is not installed. Install faiss-cpu first.")
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
        dim = embeddings.shape[1]
        store = cls(dimension=dim, index_type="hnsw")
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = 64
        emb_f32 = np.ascontiguousarray(embeddings.astype("float32"))
        index.add(emb_f32)
        store.index = index
        return store

    def save(self, index_path: str):
        """Serializes the FAISS index to disk."""
        if self.index is None:
            raise ValueError("No index built to save.")
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

    @classmethod
    def load(cls, index_path: str, index_type: str = "flat") -> "FaissVectorStore":
        """Loads a pre-built FAISS index from disk."""
        if faiss is None:
            raise ImportError("faiss is not installed. Install faiss-cpu first.")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        index = faiss.read_index(str(index_path))
        store = cls(dimension=index.d, index_type=index_type)
        store.index = index
        if index_type == "hnsw" and hasattr(store.index, "hnsw"):
            store.index.hnsw.efSearch = 64
        return store

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes an inner product search for query_vector.
        Returns:
            scores: float array of cosine similarity scores (top_k,)
            indices: int array of vector indices (top_k,)
        """
        if self.index is None:
            raise ValueError("Index not loaded or built.")
        
        q = np.ascontiguousarray(query_vector.astype("float32"))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        
        scores, indices = self.index.search(q, top_k)
        return scores[0], indices[0]

    def get_stats(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns diagnostic metadata about the index."""
        if self.index is None:
            return {"status": "uninitialized"}
        
        size_bytes = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
        return {
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "is_trained": self.index.is_trained,
            "size_mb": round(size_bytes / (1024 * 1024), 2)
        }
