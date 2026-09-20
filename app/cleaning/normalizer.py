"""
Text normalization and cleaning pipeline.
Preserves raw_text and generates clean_text for downstream NLP and conversational AI.
"""
import re
import html

# Regex patterns for normalization
CUSTOMER_HANDLE_PATTERN = re.compile(r"@\d+")
SPOTIFY_HANDLE_PATTERN = re.compile(r"@spotifycares", re.IGNORECASE)
OTHER_MENTION_PATTERN = re.compile(r"@([A-Za-z0-9_]+)")

# URL categorization patterns
SPOTIFY_HELP_URL_PATTERN = re.compile(r"https?://(?:[a-zA-Z0-9.-]+\.)?spotify\.com/\S+", re.IGNORECASE)
APP_STORE_URL_PATTERN = re.compile(r"https?://(?:itunes\.apple\.com|play\.google\.com)/\S+", re.IGNORECASE)
GENERIC_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Agent corporate signature initials at end of tweet (e.g., ^JK, -AA, ^Osebi, (1/2))
SIGNATURE_PATTERN = re.compile(r"(\s*[\^–-][A-Za-z]{2,4}\s*(\(\d+/\d+\))?|\s*\^\w+|\s*-[A-Za-z]{2,3})\s*$", re.IGNORECASE)
TWEET_NUMBERING_PATTERN = re.compile(r"\s*\(\d+/\d+\)\s*$", re.IGNORECASE)

# Multiple whitespace
WHITESPACE_PATTERN = re.compile(r"\s+")

def clean_text(raw_text: str) -> str:
    """
    Transforms raw social tweet text into clean, normalized conversational text.
    
    Processing Steps:
    1. HTML entity unescaping (&amp; -> &, etc.)
    2. URL semantic categorization (<LINK:HELP_PORTAL>, <LINK:APP_STORE>, <LINK:URL>)
    3. User handle anonymization (@12345 -> @customer, @SpotifyCares preserved)
    4. Corporate representative signoff removal (^JK, -AA)
    5. Whitespace normalization
    
    Args:
        raw_text: The original immutable tweet string.
        
    Returns:
        clean_text: The normalized string representation.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # Step 1: Decode HTML entities
    text = html.unescape(raw_text)

    # Step 2: Categorize URLs before handle masking
    text = SPOTIFY_HELP_URL_PATTERN.sub("<LINK:HELP_PORTAL>", text)
    text = APP_STORE_URL_PATTERN.sub("<LINK:APP_STORE>", text)
    text = GENERIC_URL_PATTERN.sub("<LINK:URL>", text)

    # Step 3: Handle normalization
    # Standardize Spotify handle
    text = SPOTIFY_HANDLE_PATTERN.sub("@SpotifyCares", text)
    # Mask anonymized numeric customer handles
    text = CUSTOMER_HANDLE_PATTERN.sub("@customer", text)
    # Mask any other third-party user mentions
    def _replace_other_mentions(match):
        handle = match.group(0)
        if handle.lower() == "@spotifycares" or handle.lower() == "@customer":
            return handle
        return "@mention"
    text = OTHER_MENTION_PATTERN.sub(_replace_other_mentions, text)

    # Step 4: Remove corporate representative signoff initials
    text = SIGNATURE_PATTERN.sub("", text)
    text = TWEET_NUMBERING_PATTERN.sub("", text)

    # Step 5: Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text
