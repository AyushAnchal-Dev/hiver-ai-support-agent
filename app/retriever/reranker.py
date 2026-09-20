"""
Intent-Aware Hybrid Reranker.
Fuses dense semantic search and BM25 lexical search using Reciprocal Rank Fusion (RRF)
and prioritizes candidates aligned with the incoming query's predicted intent.
"""
import math
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

class IntentAwareReranker:
    """Fuses semantic and lexical retrieval candidates with intent-aware prioritization."""

    def __init__(
        self,
        rrf_k: int = 60,
        weight_semantic: float = 1.0,
        weight_lexical: float = 0.85,
        intent_alignment_bonus: float = 0.30
    ):
        self.rrf_k = rrf_k
        self.weight_semantic = weight_semantic
        self.weight_lexical = weight_lexical
        self.intent_alignment_bonus = intent_alignment_bonus

    def rerank(
        self,
        semantic_results: List[Tuple[str, float, int]],  # (doc_id, cosine_sim, idx)
        lexical_results: List[Tuple[str, float, int]],   # (doc_id, bm25_score, idx)
        metadata_map: Dict[str, Dict[str, Any]],         # doc_id -> metadata (including 'intent')
        query_intent: Optional[str] = None,
        query_subintent: Optional[str] = None,
        top_k: int = 5,
        use_intent_bonus: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Merges, scores, and reranks candidates.
        Supports intent and subtype-aware prioritization.
        """
        candidate_scores = defaultdict(lambda: {
            "rrf_score": 0.0,
            "semantic_score": 0.0,
            "lexical_score": 0.0,
            "semantic_rank": None,
            "lexical_rank": None
        })

        # 1. Process Semantic Candidates
        for rank, (doc_id, sim, idx) in enumerate(semantic_results, start=1):
            entry = candidate_scores[doc_id]
            entry["semantic_score"] = float(sim)
            entry["semantic_rank"] = rank
            entry["rrf_score"] += self.weight_semantic / (self.rrf_k + rank)

        # 2. Process Lexical Candidates
        for rank, (doc_id, bm25, idx) in enumerate(lexical_results, start=1):
            entry = candidate_scores[doc_id]
            entry["lexical_score"] = float(bm25)
            entry["lexical_rank"] = rank
            entry["rrf_score"] += self.weight_lexical / (self.rrf_k + rank)

        # 3. Apply Intent & Subtype-Aware Alignment Bonus & Calibrate Confidence
        ranked_items = []
        for doc_id, scores in candidate_scores.items():
            meta = metadata_map.get(doc_id, {})
            cand_intent = meta.get("intent", "UNKNOWN_OTHER")
            cand_summary = (meta.get("customer_issue_summary", "") + " " + meta.get("resolution_summary", "")).lower()

            is_intent_aligned = False
            if query_intent and query_intent not in ["UNKNOWN_OTHER", "UNKNOWN_FALLBACK"]:
                if cand_intent == query_intent:
                    is_intent_aligned = True

            # Check subtype alignment for ACCOUNT_ACCESS_AUTH
            is_subintent_aligned = False
            if is_intent_aligned and query_subintent and cand_intent == "ACCOUNT_ACCESS_AUTH":
                if query_subintent == "TRAVEL_RESTRICTION" and any(k in cand_summary for k in ["14 days", "abroad", "travel", "country", "kenya", "trip"]):
                    is_subintent_aligned = True
                elif query_subintent == "PASSWORD_RESET" and any(k in cand_summary for k in ["password", "reset", "forgot"]):
                    is_subintent_aligned = True
                elif query_subintent == "LOGIN_FAILED" and any(k in cand_summary for k in ["login", "connect", "access", "failed", "unavailable"]):
                    is_subintent_aligned = True
                elif query_subintent == "ACCOUNT_COMPROMISE" and any(k in cand_summary for k in ["hacked", "stolen", "unauthorized", "someone"]):
                    is_subintent_aligned = True

            # Raw RRF score
            rrf_base = scores["rrf_score"]

            # Final ranking score: Pure RRF vs. RRF + Intent/Subtype Bonus
            ranking_score = rrf_base
            if use_intent_bonus and is_intent_aligned:
                ranking_score += (self.intent_alignment_bonus / self.rrf_k)
                if is_subintent_aligned:
                    ranking_score += (0.15 / self.rrf_k)

            # Calibrated retrieval confidence bounded in [0.0, 1.0]
            sem_norm = max(0.0, min(1.0, scores["semantic_score"]))
            lex_norm = 1.0 - math.exp(-max(0.0, scores["lexical_score"]) / 10.0)
            intent_val = 1.0 if is_intent_aligned else 0.0

            confidence = 0.50 * sem_norm + 0.30 * lex_norm + 0.20 * intent_val
            bounded_confidence = round(min(1.0, max(0.0, confidence)), 4)

            # Similarity score for legacy compatibility (cosine similarity if available, else normalized proxy)
            if scores["semantic_score"] > 0:
                sim_score = round(scores["semantic_score"], 4)
            else:
                sim_score = round(0.50 * lex_norm, 4)

            ranked_items.append({
                "thread_id": doc_id,
                "ranking_score": ranking_score,
                "similarity_score": sim_score,
                "semantic_score": round(scores["semantic_score"], 4),
                "lexical_score": round(scores["lexical_score"], 4),
                "rrf_score": round(rrf_base, 5),
                "intent_match": is_intent_aligned,
                "intent_aligned": is_intent_aligned,  # alias for backward compatibility
                "retrieval_confidence": bounded_confidence,
                "intent": cand_intent,
                "metadata": meta
            })

        # 4. Sort strictly by ranking_score descending
        ranked_items.sort(key=lambda x: x["ranking_score"], reverse=True)
        return ranked_items[:top_k]
