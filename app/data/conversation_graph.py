"""
Conversation Thread Graph Reconstruction & Topological Analysis.
Reconstructs multi-turn conversational trees using parent pointers and children adjacency maps.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
from ..schemas.models import Turn, ConversationThread
from ..cleaning.normalizer import clean_text
from ..cleaning.filters import contains_dm_deflection

def parse_twitter_timestamp(ts_str: str) -> Optional[datetime]:
    """Parses Twitter timestamp format 'Tue Oct 31 22:10:47 +0000 2017'."""
    try:
        return datetime.strptime(ts_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return None

def reconstruct_conversation_threads(
    tweets_dict: Dict[str, Dict[str, Any]],
    target_roots: Set[str],
    children_map: Dict[str, List[str]],
    brand_handle: str = "SpotifyCares"
) -> List[ConversationThread]:
    """
    Reconstructs complete conversation threads from target root tweet IDs.
    
    Args:
        tweets_dict: Dictionary mapping tweet_id -> raw tweet dictionary.
        target_roots: Set of root tweet IDs defining conversation trees.
        children_map: Adjacency list mapping parent_tweet_id -> list of child_tweet_ids.
        brand_handle: Brand name identifier.
        
    Returns:
        List of ConversationThread domain models.
    """
    conversations: List[ConversationThread] = []

    for root_id in target_roots:
        if root_id not in tweets_dict:
            continue
        
        root_data = tweets_dict[root_id]
        customer_id = root_data["author_id"] if root_data["inbound"] else "customer_unknown"
        created_at_root = root_data["created_at"]

        # Traverse the tree rooted at root_id (BFS)
        queue: List[Tuple[str, int]] = [(root_id, 1)]
        raw_turns: List[Tuple[Dict[str, Any], int]] = []
        visited: Set[str] = set()
        max_depth = 1

        while queue:
            curr_id, depth = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)

            if depth > max_depth:
                max_depth = depth

            curr_tweet = tweets_dict.get(curr_id)
            if curr_tweet:
                raw_turns.append((curr_tweet, depth))
                for ch_id in children_map.get(curr_id, []):
                    if ch_id not in visited and ch_id in tweets_dict:
                        queue.append((ch_id, depth + 1))

        # Sort turns chronologically by timestamp (fallback to turn depth)
        def _sort_key(item):
            t_data, depth = item
            dt = parse_twitter_timestamp(t_data["created_at"])
            ts = dt.timestamp() if dt else 0
            return (ts, depth)

        sorted_turns = sorted(raw_turns, key=_sort_key)

        turns: List[Turn] = []
        cust_turns = 0
        brand_turns = 0
        has_dm_deflection = False

        for idx, (t_data, depth) in enumerate(sorted_turns, start=1):
            is_inbound = t_data["inbound"]
            role = "customer" if is_inbound else "brand"
            if is_inbound:
                cust_turns += 1
                if customer_id == "customer_unknown":
                    customer_id = t_data["author_id"]
            else:
                brand_turns += 1
                if contains_dm_deflection(t_data["text"]):
                    has_dm_deflection = True

            raw_txt = t_data["text"]
            cln_txt = clean_text(raw_txt)

            turn = Turn(
                turn_id=idx,
                tweet_id=t_data["tweet_id"],
                author_id=t_data["author_id"],
                role=role,
                created_at=t_data["created_at"],
                raw_text=raw_txt,
                clean_text=cln_txt,
                in_response_to_tweet_id=t_data.get("in_response_to_tweet_id") or None,
                turn_depth=depth
            )
            turns.append(turn)

        # Derive thread operational status
        if brand_turns == 0:
            status = "unanswered"
        elif has_dm_deflection:
            status = "deflected_to_dm"
        else:
            status = "resolved_publicly"

        thread = ConversationThread(
            thread_id=root_id,
            root_tweet_id=root_id,
            customer_id=customer_id,
            brand=brand_handle,
            status=status,
            created_at=created_at_root,
            conversation_length=len(turns),
            maximum_depth=max_depth,
            customer_turns=cust_turns,
            brand_turns=brand_turns,
            turns=turns
        )
        conversations.append(thread)

    return conversations
