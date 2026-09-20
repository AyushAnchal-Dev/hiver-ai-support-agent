"""
Production Retriever Agent.
Implements hybrid retrieval (Semantic dense search + BM25 lexical search + Intent-aware reranker).
Returns top-5 grounded customer support conversations with dialogue turns and metadata.
"""
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from app.retriever.semantic_search import SemanticSearch
from app.retriever.lexical_search import BM25LexicalSearch
from app.retriever.reranker import IntentAwareReranker
from app.agents.intent_agent import IntentAgent

@dataclass
class RetrievedConversation:
    """Output contract for retrieved past support conversation."""
    thread_id: str
    intent: str
    similarity_score: float
    customer_issue_summary: str
    resolution_summary: str
    retrieved_turns: List[Dict[str, Any]]
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    rrf_score: float = 0.0
    intent_match: bool = False
    retrieval_confidence: float = 0.0
    intent_aligned: bool = False  # backward compatibility alias
    resolution_type: str = "direct_resolution"
    contains_dm_request: bool = False
    contains_link: bool = False
    ranking_score: float = 0.0

class RetrieverAgent:
    """Production hybrid retrieval agent for Spotify customer support."""

    index_loaded: bool = False
    index_load_count: int = 0
    _shared_semantic_retriever: Optional[SemanticSearch] = None
    _shared_metadata_map: Dict[str, Dict[str, Any]] = {}
    _shared_doc_ids: List[str] = []
    _shared_conversations_by_id: Dict[str, List[Dict[str, Any]]] = {}
    _shared_lexical_retriever: Optional[BM25LexicalSearch] = None
    _shared_reranker: Optional[IntentAwareReranker] = None
    _shared_intent_agent: Optional[IntentAgent] = None

    def __init__(
        self,
        corpus_path: Optional[str] = None,
        index_path: Optional[str] = None,
        index_type: str = "flat",
        conversations_json_path: Optional[str] = None,
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.corpus_path = corpus_path or str(root_dir / "data" / "thread_corpus.csv")
        self.index_path = index_path or str(root_dir / "data" / f"faiss_{index_type}.index")
        self.index_type = index_type
        self.conversations_json_path = conversations_json_path or str(root_dir / "data" / "spotify_conversations.json")
        self.model_name = model_name

        # State storage
        self.metadata_map: Dict[str, Dict[str, Any]] = {}
        self.doc_ids: List[str] = []
        self.conversations_by_id: Dict[str, List[Dict[str, Any]]] = {}

        # Sub-components
        self.intent_agent: Optional[IntentAgent] = None
        self.semantic_retriever: Optional[SemanticSearch] = None
        self.lexical_retriever: Optional[BM25LexicalSearch] = None
        self.reranker: Optional[IntentAwareReranker] = None

        self._initialized = False

    @classmethod
    def reset_index(cls):
        """Resets singleton state (for isolated unit testing)."""
        cls.index_loaded = False
        cls.index_load_count = 0
        cls._shared_semantic_retriever = None
        cls._shared_metadata_map = {}
        cls._shared_doc_ids = []
        cls._shared_conversations_by_id = {}
        cls._shared_lexical_retriever = None
        cls._shared_reranker = None
        cls._shared_intent_agent = None

    def initialize(self):
        """Loads corpus, builds lexical index, and initializes sub-components."""
        if self._initialized:
            return

        # Reuse shared singleton if already loaded in process
        if RetrieverAgent.index_loaded and RetrieverAgent._shared_semantic_retriever is not None:
            self.metadata_map = RetrieverAgent._shared_metadata_map
            self.doc_ids = RetrieverAgent._shared_doc_ids
            self.conversations_by_id = RetrieverAgent._shared_conversations_by_id
            self.intent_agent = RetrieverAgent._shared_intent_agent
            self.semantic_retriever = RetrieverAgent._shared_semantic_retriever
            self.lexical_retriever = RetrieverAgent._shared_lexical_retriever
            self.reranker = RetrieverAgent._shared_reranker
            self._initialized = True
            return

        # 1. Load corpus
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        df_corpus = pd.read_csv(self.corpus_path)
        self.doc_ids = [str(tid) for tid in df_corpus["thread_id"]]

        for _, row in df_corpus.iterrows():
            tid = str(row["thread_id"])
            self.metadata_map[tid] = {
                "thread_id": tid,
                "intent": str(row.get("intent", "UNKNOWN_OTHER")),
                "customer_issue_summary": str(row.get("customer_issue_summary", "")),
                "resolution_summary": str(row.get("resolution_summary", "")),
                "conversation_length": int(row.get("conversation_length", 1)),
                "customer_turns": int(row.get("customer_turns", 1)),
                "brand_turns": int(row.get("brand_turns", 0)),
                "resolution_type": str(row.get("resolution_type", "direct_resolution")),
                "contains_dm_request": bool(row.get("contains_dm_request", False)),
                "contains_link": bool(row.get("contains_link", False))
            }

        # 2. Load conversations for full turn history
        if os.path.exists(self.conversations_json_path):
            with open(self.conversations_json_path, "r", encoding="utf-8") as f:
                raw_convs = json.load(f)
            for c in raw_convs:
                self.conversations_by_id[str(c["thread_id"])] = c.get("turns", [])

        # 3. Initialize Intent Agent
        self.intent_agent = IntentAgent()

        # 4. Initialize Semantic Retriever
        self.semantic_retriever = SemanticSearch(
            model_name=self.model_name,
            index_path=self.index_path if os.path.exists(self.index_path) else None,
            index_type=self.index_type,
            doc_ids=self.doc_ids
        )

        # 5. Initialize & Fit BM25 Lexical Retriever
        self.lexical_retriever = BM25LexicalSearch()
        corpus_texts = df_corpus["conversation_text"].fillna("").tolist()
        self.lexical_retriever.fit(corpus_texts, doc_ids=self.doc_ids)

        # 6. Initialize Reranker
        self.reranker = IntentAwareReranker()

        # Cache shared singletons
        RetrieverAgent._shared_metadata_map = self.metadata_map
        RetrieverAgent._shared_doc_ids = self.doc_ids
        RetrieverAgent._shared_conversations_by_id = self.conversations_by_id
        RetrieverAgent._shared_intent_agent = self.intent_agent
        RetrieverAgent._shared_semantic_retriever = self.semantic_retriever
        RetrieverAgent._shared_lexical_retriever = self.lexical_retriever
        RetrieverAgent._shared_reranker = self.reranker

        RetrieverAgent.index_loaded = True
        RetrieverAgent.index_load_count += 1
        self._initialized = True


    def retrieve(
        self,
        customer_message: str,
        top_k: int = 5,
        candidate_k: int = 20,
        use_intent_bonus: bool = True
    ) -> List[RetrievedConversation]:
        """
        Executes hybrid retrieval:
        1. Predict query intent via IntentAgent.
        2. Retrieve Top-20 dense semantic candidates.
        3. Retrieve Top-20 BM25 lexical candidates.
        4. Fuse candidates via RRF (optionally applying intent-alignment bonus).
        5. Return Top-5 RetrievedConversation objects.
        """
        if not self._initialized:
            self.initialize()

        if not customer_message or not customer_message.strip():
            return []

        # 1. Triage query intent
        intent_pred = self.intent_agent.predict(customer_message)
        query_intent = intent_pred.predicted_intent

        # 2. Semantic Search (Top-20)
        semantic_candidates = []
        if self.semantic_retriever and self.semantic_retriever.vector_store:
            try:
                semantic_candidates = self.semantic_retriever.search(customer_message, top_k=candidate_k)
            except Exception:
                semantic_candidates = []

        # 3. Lexical Search (Top-20)
        lexical_candidates = []
        if self.lexical_retriever:
            lexical_candidates = self.lexical_retriever.search(customer_message, top_k=candidate_k)

        # Check subintent for ACCOUNT_ACCESS_AUTH
        query_subintent = None
        if query_intent == "ACCOUNT_ACCESS_AUTH":
            cm_low = customer_message.lower()
            if any(w in cm_low for w in ["hacked", "someone logged in", "unauthorized", "changed email", "stolen", "someone's using"]):
                query_subintent = "ACCOUNT_COMPROMISE"
            elif any(w in cm_low for w in ["abroad", "14 days", "country restriction", "travelling", "traveling", "kenya", "trip"]):
                query_subintent = "TRAVEL_RESTRICTION"
            elif any(w in cm_low for w in ["forgot password", "reset password", "reset email", "change password", "new password"]):
                query_subintent = "PASSWORD_RESET"
            elif any(w in cm_low for w in ["login unavailable", "page not found", "login failed", "cannot login", "can't login", "unable to login", "can't access", "login"]):
                query_subintent = "LOGIN_FAILED"

        # 4. Hybrid Reranking (Top-5)
        reranked = self.reranker.rerank(
            semantic_results=semantic_candidates,
            lexical_results=lexical_candidates,
            metadata_map=self.metadata_map,
            query_intent=query_intent,
            query_subintent=query_subintent,
            top_k=top_k,
            use_intent_bonus=use_intent_bonus
        )

        # 5. Pack into RetrievedConversation output objects
        results = []
        for r in reranked:
            tid = r["thread_id"]
            meta = r["metadata"]
            turns = self.conversations_by_id.get(tid, [])

            results.append(RetrievedConversation(
                thread_id=tid,
                intent=r["intent"],
                similarity_score=r["similarity_score"],
                customer_issue_summary=meta.get("customer_issue_summary", ""),
                resolution_summary=meta.get("resolution_summary", ""),
                retrieved_turns=turns,
                semantic_score=r.get("semantic_score", 0.0),
                lexical_score=r.get("lexical_score", 0.0),
                rrf_score=r.get("rrf_score", 0.0),
                intent_match=r.get("intent_match", False),
                retrieval_confidence=r.get("retrieval_confidence", 0.0),
                intent_aligned=r.get("intent_aligned", False),
                resolution_type=meta.get("resolution_type", "direct_resolution"),
                contains_dm_request=meta.get("contains_dm_request", False),
                contains_link=meta.get("contains_link", False),
                ranking_score=r.get("ranking_score", 0.0)
            ))

        return results
