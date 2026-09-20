"""
Memory-efficient data loading and persistence utilities.
"""
import os
import csv
import json
from typing import Dict, List, Set, Tuple, Any, Generator
from ..schemas.models import ConversationThread, DatasetMetadata

def stream_csv_rows(csv_path: str) -> Generator[Dict[str, Any], None, None]:
    """Streams rows from TWCS CSV as parsed dictionaries with downcasted booleans."""
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            yield {
                "tweet_id": row[0],
                "author_id": row[1],
                "inbound": (row[2] == "True"),
                "created_at": row[3],
                "text": row[4],
                "response_tweet_id": row[5] if row[5] else None,
                "in_response_to_tweet_id": row[6] if row[6] else None
            }

def save_conversations_json(conversations: List[ConversationThread], json_path: str):
    """Saves a list of ConversationThread objects to hierarchical JSON."""
    data = [c.to_dict() for c in conversations]
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_metadata_json(metadata: DatasetMetadata, json_path: str):
    """Saves DatasetMetadata to JSON."""
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
