"""
Baseline 1: Rule-based intent classifier using regex patterns and keyword matching.
"""
import re
from typing import Dict, Any, Tuple, Optional

INTENT_RULES = {
    "ACCOUNT_ACCESS_AUTH": [
        r"\b(log\s*in|login|logged out|password|reset|hacked|compromised|stolen account|wrong credentials|recover account|can't get in|access my account)\b",
        r"\b(change email|verify email|authentication|auth|verification code|locked out)\b"
    ],
    "SUBSCRIPTION_BILLING": [
        r"\b(charged|charge|bill|billing|subscription|premium|receipt|refund|double charge|unauthorized charge)\b",
        r"\b(credit card|debit card|paypal|student discount|sheerid|family plan|family account|payment failed|upgrade|cancel subscription)\b"
    ],
    "PLAYBACK_STREAMING": [
        r"\b(play|playing|pause|pauses|pausing|skip|skipping|stutter|stuttering|buffer|buffering|stops playing|cuts out|won't play)\b",
        r"\b(audio quality|distort|distortion|sound crackling|no sound|volume too low|playback error)\b"
    ],
    "OFFLINE_DOWNLOAD_SYNC": [
        r"\b(offline|download|downloads|downloaded|downloading|sd card|cache|storage|clear cache|airplane mode|offline sync)\b"
    ],
    "PLAYLIST_LIBRARY_MGMT": [
        r"\b(playlist|playlists|library|saved songs|liked songs|discover weekly|release radar|daily mix|queue|songs disappeared|lost my playlist|recover playlist)\b"
    ],
    "APP_CRASH_TECHNICAL_BUG": [
        r"\b(crash|crashes|crashing|freeze|freezes|freezing|black screen|blank screen|not opening|force close|won't open|app bug|glitch)\b",
        r"\b(after update|new update broke|app unresponsive|fatal error)\b"
    ],
    "DEVICE_CONNECT_INTEGRATION": [
        r"\b(spotify connect|connect device|bluetooth|carplay|android auto|alexa|echo|sonos|chromecast|ps4|xbox|web player|browser playback|smart tv|airplay)\b"
    ],
    "CONTENT_METADATA_AVAILABILITY": [
        r"\b(greyed out|grayed out|not available|missing song|missing album|artist missing|lyrics|explicit filter|clean version|licensing rights|regional restriction)\b",
        r"\b(isn't on spotify|not on spotify|removed from spotify)\b"
    ],
    "ADS_PROMOTIONAL_ISSUES": [
        r"\b(ad|ads|advert|advertisement|commercial|commercials|banner|pop up|audio ad)\b",
        r"\b(ads on premium|paying for premium but getting ads|repetitive ad|inappropriate ad)\b"
    ],
    "UI_FEEDBACK_FEATURE_REQUEST": [
        r"\b(feature request|bring back|why did you remove|hate the new|redesign|widget|interface change|ui suggestion|new layout)\b"
    ]
}

class RuleBasedClassifier:
    """Deterministic rule-based baseline classifier."""

    def __init__(self):
        self.compiled_rules = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_RULES.items()
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict intent by counting matching rule regex patterns.
        """
        if not text or not isinstance(text, str):
            return {
                "predicted_intent": "UNKNOWN_OTHER",
                "confidence": 0.0,
                "match_count": 0
            }

        scores: Dict[str, int] = {}
        for intent, patterns in self.compiled_rules.items():
            count = sum(len(p.findall(text)) for p in patterns)
            if count > 0:
                scores[intent] = count

        if not scores:
            # Check greeting
            if re.search(r"\b(hello|hey|hi|good\s+morning|good\s+evening|thanks|thank\s+you|check\s+dm|dm\s+sent)\b", text, re.IGNORECASE):
                return {"predicted_intent": "GREETING_OR_CHAT", "confidence": 0.80, "match_count": 1, "all_matches": {}}
            # Check general complaint
            if re.search(r"\b(sucks|terrible|worst|horrible|trash|garbage|awful|useless|hate|broken|fix\s+this|disappointed)\b", text, re.IGNORECASE):
                return {"predicted_intent": "GENERAL_COMPLAINT", "confidence": 0.75, "match_count": 1, "all_matches": {}}
            return {
                "predicted_intent": "UNKNOWN_OTHER",
                "confidence": 0.20,
                "match_count": 0
            }

        # Sort by match count
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_count = sorted_scores[0]
        total_matches = sum(scores.values())

        confidence = round(min(0.95, 0.40 + 0.20 * top_count), 2)
        if len(sorted_scores) > 1 and sorted_scores[0][1] == sorted_scores[1][1]:
            confidence = 0.50  # Tie penalty

        return {
            "predicted_intent": top_intent,
            "confidence": confidence,
            "match_count": top_count,
            "all_matches": scores
        }
