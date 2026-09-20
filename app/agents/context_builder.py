"""
Context Builder & Compression Layer.
Distills Top-5 retrieved multi-turn conversations into concise, de-duplicated evidence snippets.
Enforces character/token budgets and preserves exact thread_id and turn_id citations.
The Resolver Agent receives ONLY compressed EvidenceSnippet objects, never raw full conversations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
import re

from app.agents.retriever_agent import RetrievedConversation

@dataclass
class EvidenceSnippet:
    """A compressed, verified evidence unit from past support interactions."""
    snippet_id: str          # e.g. 'T199189-Turn2'
    thread_id: str
    turn_id: int
    role: str                # 'customer' or 'brand'
    intent: str
    content: str             # Clean, condensed message snippet
    snippet_type: str        # 'symptom' or 'resolution'
    char_count: int = 0

    def __post_init__(self):
        if not self.char_count:
            self.char_count = len(self.content)

@dataclass
class CompressedContext:
    """Structured, compressed context package passed to ResolverAgent."""
    snippets: List[EvidenceSnippet]
    cited_thread_ids: List[str]
    total_chars: int
    estimated_tokens: int
    has_dm_deflection: bool
    has_link: bool
    top_intent: str
    mean_retrieval_confidence: float

class ContextBuilder:
    """Compresses and filters retrieved conversation history into actionable evidence."""

    def __init__(self, max_chars: int = 2500, max_snippets: int = 8, cache_size: int = 512):
        self.max_chars = max_chars
        self.max_snippets = max_snippets
        self._cache_size = cache_size
        self._cache: Dict[Tuple[str, ...], CompressedContext] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 4) if total > 0 else 0.0

    def get_cache_hit_rate(self) -> float:
        return self.cache_hit_rate

    def clear_cache(self):
        """Clears the memoization cache and resets counters."""
        self._cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes excessive whitespace and clean customer handles."""
        t = re.sub(r"\s+", " ", text or "").strip()
        return t

    @staticmethod
    def _is_noisy_turn(text: str, role: str) -> bool:
        """Flags empty, purely repetitive, or uninformative chatter turns."""
        t_low = text.lower()
        if len(t_low) < 8:
            return True
        # Purely social acknowledgements without advice
        if role == "brand" and t_low in ["you're welcome! /gk", "no problem! /gk", "glad to help!", "thanks! /gk"]:
            return True
        return False

    @staticmethod
    def _compute_jaccard_similarity(s1: str, s2: str) -> float:
        """Word-level Jaccard similarity for deduplication."""
        words1 = set(re.findall(r"\w+", s1.lower()))
        words2 = set(re.findall(r"\w+", s2.lower()))
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def build_context(
        self,
        retrieved_conversations: List[RetrievedConversation],
        query_intent: Optional[str] = None
    ) -> CompressedContext:
        """
        Compresses Top-5 retrieved conversations:
        1. Checks memoization cache using tuple of thread IDs.
        2. Filters noise and extracts distinct symptom and resolution turns.
        3. Deduplicates overlapping advice across threads.
        4. Enforces token/character budget limit.
        """
        if retrieved_conversations:
            cache_key = tuple(str(conv.thread_id) for conv in retrieved_conversations)
            if cache_key in self._cache:
                self.cache_hits += 1
                return self._cache[cache_key]
            self.cache_misses += 1
        else:
            cache_key = None

        accepted_snippets: List[EvidenceSnippet] = []
        current_chars = 0
        accepted_resolutions: List[str] = []
        confidences: List[float] = []
        has_dm = False
        has_link = False

        # Process retrieved conversations in strict priority order (highest retrieval score first)
        for conv in retrieved_conversations:
            if len(accepted_snippets) >= self.max_snippets:
                break

            tid = str(conv.thread_id)
            intent = conv.intent
            confidences.append(conv.retrieval_confidence)
            if conv.contains_dm_request:
                has_dm = True
            if conv.contains_link:
                has_link = True

            # Extract at most 1 symptom turn (customer) and at most 1 resolution turn (brand)
            best_symptom: Optional[EvidenceSnippet] = None
            best_resolution: Optional[EvidenceSnippet] = None

            for turn in conv.retrieved_turns:
                turn_id = turn.get("turn_id", 1)
                role = turn.get("role", "brand" if not turn.get("is_inbound", True) else "customer")
                text = self._clean_text(turn.get("clean_text", ""))

                if self._is_noisy_turn(text, role):
                    continue

                if role == "customer" and best_symptom is None:
                    best_symptom = EvidenceSnippet(
                        snippet_id=f"T{tid}-Turn{turn_id}",
                        thread_id=tid,
                        turn_id=turn_id,
                        role="customer",
                        intent=intent,
                        content=text,
                        snippet_type="symptom"
                    )
                elif role == "brand" and best_resolution is None:
                    best_resolution = EvidenceSnippet(
                        snippet_id=f"T{tid}-Turn{turn_id}",
                        thread_id=tid,
                        turn_id=turn_id,
                        role="brand",
                        intent=intent,
                        content=text,
                        snippet_type="resolution"
                    )

            # Priority: 1. Symptom first
            if best_symptom:
                if len(accepted_snippets) < self.max_snippets and (current_chars + best_symptom.char_count <= self.max_chars):
                    accepted_snippets.append(best_symptom)
                    current_chars += best_symptom.char_count

            # Priority: 2. Matching brand reply second (with Jaccard deduplication across threads)
            if best_resolution:
                is_duplicate = False
                for acc_res in accepted_resolutions:
                    if self._compute_jaccard_similarity(best_resolution.content, acc_res) > 0.60:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    if len(accepted_snippets) < self.max_snippets and (current_chars + best_resolution.char_count <= self.max_chars):
                        accepted_snippets.append(best_resolution)
                        accepted_resolutions.append(best_resolution.content)
                        current_chars += best_resolution.char_count

        cited_tids = sorted(list({s.thread_id for s in accepted_snippets}))
        mean_conf = round(float(sum(confidences) / len(confidences)), 4) if confidences else 0.0
        top_intent = query_intent or (retrieved_conversations[0].intent if retrieved_conversations else "UNKNOWN_OTHER")

        compressed = CompressedContext(
            snippets=accepted_snippets,
            cited_thread_ids=cited_tids,
            total_chars=current_chars,
            estimated_tokens=int(current_chars / 4.2),
            has_dm_deflection=has_dm,
            has_link=has_link,
            top_intent=top_intent,
            mean_retrieval_confidence=mean_conf
        )

        if cache_key is not None:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = compressed

        return compressed

