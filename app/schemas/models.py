"""
Data models and schemas for conversations, turns, and dataset metadata.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

@dataclass
class Turn:
    turn_id: int
    tweet_id: str
    author_id: str
    role: str  # "customer" or "brand"
    created_at: str
    raw_text: str
    clean_text: str
    in_response_to_tweet_id: Optional[str] = None
    turn_depth: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConversationThread:
    thread_id: str
    root_tweet_id: str
    customer_id: str
    brand: str
    status: str  # "resolved_publicly", "deflected_to_dm", "unanswered"
    created_at: str
    conversation_length: int
    maximum_depth: int
    customer_turns: int
    brand_turns: int
    turns: List[Turn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["turns"] = [t.to_dict() if isinstance(t, Turn) else t for t in self.turns]
        return data

@dataclass
class DatasetMetadata:
    source_dataset: str
    target_brand: str
    total_tweets: int
    total_threads: int
    average_thread_length: float
    median_thread_length: float
    maximum_depth: int
    generation_timestamp: str
    schema_version: str = "2.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
