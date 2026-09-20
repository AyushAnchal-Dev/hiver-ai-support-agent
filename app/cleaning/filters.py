"""
Quality filters and text classification heuristics.
"""
import re

DM_KEYWORDS = re.compile(r"\b(dm|direct message|private message)\b", re.IGNORECASE)
PROFANITY_PATTERN = re.compile(r"\b(fuck|shit|damn|bitch|crap|piss|asshole|hell|suck|horrible|terrible|worst|trash)\b", re.IGNORECASE)

def contains_dm_deflection(text: str) -> bool:
    """Returns True if message deflects interaction to private messaging (DM)."""
    if not text:
        return False
    return bool(DM_KEYWORDS.search(text))

def contains_profanity(text: str) -> bool:
    """Returns True if message contains toxic profanity or anger keywords."""
    if not text:
        return False
    return bool(PROFANITY_PATTERN.search(text))

def is_informative_turn(text: str, min_words: int = 3) -> bool:
    """Returns True if message has sufficient length and content."""
    if not text:
        return False
    return len(text.split()) >= min_words
