"""
Production Intent Classification Agent.
Implements a deterministic multi-feature intent triage pipeline for Spotify customer support.
"""
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from app.agents.interfaces import ITriageAgent, TriageResult
from app.schemas.models import Turn
from app.cleaning.normalizer import clean_text
from app.cleaning.filters import contains_profanity

@dataclass
class IntentDefinition:
    intent_id: str
    intent_name: str
    priority: str  # "high", "medium", "low"
    auto_handle_candidate: bool
    core_regexes: List[re.Pattern]
    keywords: List[str]
    exclusions: List[re.Pattern] = field(default_factory=list)

@dataclass
class IntentPrediction:
    predicted_intent: str
    intent_id: str
    confidence: float
    priority: str
    auto_handle_candidate: bool
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    candidate_scores: Dict[str, float] = field(default_factory=dict)

# ==============================================================================
# Intent Taxonomy Definitions & Multi-Feature Rules
# ==============================================================================

TAXONOMY_SPECS: List[Dict[str, Any]] = [
    {
        "intent_id": "INT_01",
        "intent_name": "ACCOUNT_ACCESS_AUTH",
        "priority": "high",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(log\s*in|login|logged out|password|reset|hacked|compromised|stolen account|wrong credentials|recover account|can't get in|access my account)\b",
            r"\b(change email|verify email|authentication|auth|verification code|locked out|security code|username)\b"
        ],
        "keywords": ["login", "password", "email", "reset", "hacked", "account", "logged", "auth", "credentials"],
        "exclusions": [r"\b(student discount|family plan invite|charged twice)\b"]
    },
    {
        "intent_id": "INT_02",
        "intent_name": "SUBSCRIPTION_BILLING",
        "priority": "high",
        "auto_handle_candidate": False,
        "patterns": [
            r"\b(charged|charge|bill|billing|subscription|premium|receipt|refund|double charge|unauthorized charge)\b",
            r"\b(credit card|debit card|paypal|student discount|sheerid|family plan|family account|payment failed|upgrade to premium|cancel subscription)\b"
        ],
        "keywords": ["charged", "bill", "billing", "premium", "refund", "subscription", "student", "discount", "family", "payment", "card"],
        "exclusions": [r"\b(ads on premium|ad playing)\b"]
    },
    {
        "intent_id": "INT_03",
        "intent_name": "PLAYBACK_STREAMING",
        "priority": "medium",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(play|playing|pause|pauses|pausing|skip|skipping|stutter|stuttering|buffer|buffering|stops playing|cuts out|won't play)\b",
            r"\b(audio quality|distort|distortion|sound crackling|no sound|volume too low|playback error)\b"
        ],
        "keywords": ["play", "pause", "skipping", "buffer", "buffering", "stutter", "audio", "sound", "volume", "stops"],
        "exclusions": [r"\b(offline|download|downloaded|connect to speaker|bluetooth|car)\b"]
    },
    {
        "intent_id": "INT_04",
        "intent_name": "OFFLINE_DOWNLOAD_SYNC",
        "priority": "medium",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(offline|download|downloads|downloaded|downloading|sd card|cache|storage|clear cache|airplane mode|offline sync)\b",
            r"\b(offline songs won't play|not downloading|download limit)\b"
        ],
        "keywords": ["offline", "download", "downloaded", "downloading", "cache", "storage", "sd card", "sync"],
        "exclusions": [r"\b(charged for premium)\b"]
    },
    {
        "intent_id": "INT_05",
        "intent_name": "PLAYLIST_LIBRARY_MGMT",
        "priority": "medium",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(playlist|playlists|library|saved songs|liked songs|discover weekly|release radar|daily mix|queue|songs disappeared|lost my playlist|recover playlist)\b",
            r"\b(deleted playlist|restore playlist|add to playlist|order of songs)\b"
        ],
        "keywords": ["playlist", "library", "saved", "liked", "discover weekly", "release radar", "queue", "disappeared", "restore"],
        "exclusions": [r"\b(licensing rights|not available in my country)\b"]
    },
    {
        "intent_id": "INT_06",
        "intent_name": "APP_CRASH_TECHNICAL_BUG",
        "priority": "high",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(crash|crashes|crashing|freeze|freezes|freezing|black screen|blank screen|not opening|force close|won't open|app bug|glitch)\b",
            r"\b(after update|new update broke|app unresponsive|fatal error)\b"
        ],
        "keywords": ["crash", "crashes", "crashing", "freeze", "freezing", "black screen", "close", "bug", "glitch", "unresponsive"],
        "exclusions": [r"\b(songs pause|connect speaker)\b"]
    },
    {
        "intent_id": "INT_07",
        "intent_name": "DEVICE_CONNECT_INTEGRATION",
        "priority": "medium",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(spotify connect|connect device|bluetooth|carplay|android auto|alexa|echo|sonos|chromecast|ps4|xbox|web player|browser playback|smart tv|airplay)\b",
            r"\b(can't find speaker|won't connect to|device not showing)\b"
        ],
        "keywords": ["connect", "bluetooth", "alexa", "echo", "sonos", "chromecast", "ps4", "carplay", "web player", "speaker"],
        "exclusions": [r"\b(login to account)\b"]
    },
    {
        "intent_id": "INT_08",
        "intent_name": "CONTENT_METADATA_AVAILABILITY",
        "priority": "low",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(greyed out|grayed out|not available|missing song|missing album|artist missing|lyrics|explicit filter|clean version|licensing rights|regional restriction)\b",
            r"\b(isn't on spotify|not on spotify|removed from spotify|why was song removed)\b"
        ],
        "keywords": ["greyed out", "not available", "missing", "artist", "album", "lyrics", "explicit", "removed", "region", "country"],
        "exclusions": [r"\b(playlist disappeared|library deleted)\b"]
    },
    {
        "intent_id": "INT_09",
        "intent_name": "ADS_PROMOTIONAL_ISSUES",
        "priority": "medium",
        "auto_handle_candidate": False,
        "patterns": [
            r"\b(ad|ads|advert|advertisement|commercial|commercials|banner|pop up|audio ad)\b",
            r"\b(ads on premium|paying for premium but getting ads|repetitive ad|inappropriate ad)\b"
        ],
        "keywords": ["ad", "ads", "advert", "advertisement", "commercial", "banner", "commercials"],
        "exclusions": [r"\b(charged twice)\b"]
    },
    {
        "intent_id": "INT_10",
        "intent_name": "UI_FEEDBACK_FEATURE_REQUEST",
        "priority": "low",
        "auto_handle_candidate": True,
        "patterns": [
            r"\b(feature request|bring back|why did you remove|hate the new|redesign|widget|interface change|ui suggestion|new layout)\b",
            r"\b(would love to have|please add option|new update looks awful)\b"
        ],
        "keywords": ["feature", "bring back", "remove", "widget", "redesign", "interface", "layout", "suggestion", "hate new"],
        "exclusions": [r"\b(app crashed|app won't open)\b"]
    }
]

class IntentAgent(ITriageAgent):
    """Production-grade deterministic Intent Triage Agent."""

    def __init__(self):
        self.intents: List[IntentDefinition] = []
        for spec in TAXONOMY_SPECS:
            self.intents.append(IntentDefinition(
                intent_id=spec["intent_id"],
                intent_name=spec["intent_name"],
                priority=spec["priority"],
                auto_handle_candidate=spec["auto_handle_candidate"],
                core_regexes=[re.compile(p, re.IGNORECASE) for p in spec["patterns"]],
                keywords=spec["keywords"],
                exclusions=[re.compile(e, re.IGNORECASE) for e in spec["exclusions"]]
            ))

    def predict(self, text: str) -> IntentPrediction:
        """
        Classifies an input customer text into one of the 10 Spotify intents
        with calibrated confidence and ambiguity detection.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return IntentPrediction(
                predicted_intent="UNKNOWN_OTHER",
                intent_id="INT_OTHER",
                confidence=0.0,
                priority="low",
                auto_handle_candidate=False,
                is_ambiguous=True,
                ambiguity_reason="Empty or null text input"
            )

        clean = clean_text(text)
        lower_txt = clean.lower()
        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", lower_txt))

        scores: Dict[str, float] = {}
        for intent in self.intents:
            score = 0.0

            # 1. Core Regex Matches (Strong signal: +3.0 per match)
            for reg in intent.core_regexes:
                matches = len(reg.findall(clean))
                score += matches * 3.0

            # 2. Keyword Overlap (+0.6 per matching keyword)
            for kw in intent.keywords:
                if kw in tokens or (kw in lower_txt and len(kw.split()) > 1):
                    score += 0.6

            # 3. Negative Exclusion Penalties (-2.5 per match)
            for excl in intent.exclusions:
                if excl.search(clean):
                    score -= 2.5

            scores[intent.intent_name] = max(0.0, score)

        # Disambiguation heuristics for common overlapping pairs
        # Case A: Getting ads on Premium (INT_09 vs INT_02)
        if re.search(r"\b(ads?|commercials?)\b", lower_txt) and re.search(r"\b(premium|paying|subscription)\b", lower_txt):
            scores["ADS_PROMOTIONAL_ISSUES"] += 2.0

        # Case B: Offline songs not playing (INT_04 vs INT_03)
        if re.search(r"\b(offline|download)\b", lower_txt) and re.search(r"\b(play|playing|pause)\b", lower_txt):
            scores["OFFLINE_DOWNLOAD_SYNC"] += 2.0
            scores["PLAYBACK_STREAMING"] *= 0.5

        # Check if all scores are below threshold (< 0.5) -> Decompose UNKNOWN space
        max_score = max(scores.values()) if scores else 0.0
        if max_score < 0.5:
            # 1. Check for Greeting, Social Chatter, or DM notifications
            greet_match = re.search(
                r"\b(hello|hey|hi|good\s+morning|good\s+evening|good\s+afternoon|howdy|sup|greetings|thanks|thank\s+you|thx|cheers|checking\s+in|have\s+a\s+good\s+day|can\s+someone\s+help|need\s+help|anyone\s+there|help\s+me\s+please|quick\s+question)\b|\b(dm|pm|direct\s+message|private\s+message)\b",
                lower_txt
            )
            if greet_match:
                return IntentPrediction(
                    predicted_intent="GREETING_OR_CHAT",
                    intent_id="INT_GREET",
                    confidence=0.82,
                    priority="low",
                    auto_handle_candidate=True,
                    is_ambiguous=False,
                    ambiguity_reason=None,
                    candidate_scores=scores
                )

            # 2. Check for General Non-Specific Emotional Complaint
            complaint_match = re.search(
                r"\b(sucks|terrible|worst|horrible|trash|garbage|awful|useless|hate|broken|fix\s+this|fix\s+your|fix\s+the|disappointed|annoying|annoyed|ridiculous|pathetic|waste\s+of\s+money|unacceptable|pissed|frustrated|fed\s+up|sort\s+it\s+out|not\s+working|ruined|a\s+joke|trash\s+app|garbage\s+app|worst\s+app|worst\s+service)\b",
                lower_txt
            )
            if complaint_match:
                return IntentPrediction(
                    predicted_intent="GENERAL_COMPLAINT",
                    intent_id="INT_COMPLAINT",
                    confidence=0.78,
                    priority="medium",
                    auto_handle_candidate=False,
                    is_ambiguous=False,
                    ambiguity_reason=None,
                    candidate_scores=scores
                )

            # 3. Residual Out-of-Domain or Unparseable Query
            return IntentPrediction(
                predicted_intent="UNKNOWN_OTHER",
                intent_id="INT_OTHER",
                confidence=0.25,
                priority="low",
                auto_handle_candidate=False,
                is_ambiguous=True,
                ambiguity_reason="Zero significant keyword or pattern matches across taxonomy",
                candidate_scores=scores
            )

        # Ranked intents
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_name, top_score = sorted_candidates[0]
        second_name, second_score = sorted_candidates[1]

        # Find matching definition
        top_def = next(i for i in self.intents if i.intent_name == top_name)

        # Softmax probability calibration over top candidates
        top_slice = [sc for _, sc in sorted_candidates[:4]]
        exp_sc = [math.exp(min(20.0, s * 0.8)) for s in top_slice]
        sum_exp = sum(exp_sc)
        base_confidence = exp_sc[0] / sum_exp if sum_exp > 0 else 0.50

        # Margin ambiguity check
        margin = (top_score - second_score) / (top_score + 1e-6)
        is_ambiguous = (margin < 0.25 and second_score >= 1.5) or (base_confidence < 0.70)
        ambiguity_reason = None
        if is_ambiguous:
            ambiguity_reason = f"Close competition between '{top_name}' ({top_score:.1f}) and '{second_name}' ({second_score:.1f})"

        # Bound confidence
        calibrated_confidence = round(min(0.96, max(0.30, base_confidence)), 2)

        return IntentPrediction(
            predicted_intent=top_name,
            intent_id=top_def.intent_id,
            confidence=calibrated_confidence,
            priority=top_def.priority,
            auto_handle_candidate=top_def.auto_handle_candidate,
            is_ambiguous=is_ambiguous,
            ambiguity_reason=ambiguity_reason,
            candidate_scores=scores
        )

    def triage_message(self, turn: Turn) -> TriageResult:
        """Implementation of ITriageAgent interface."""
        pred = self.predict(turn.clean_text)
        has_prof = contains_profanity(turn.clean_text)
        sentiment = -0.5 if has_prof else (-0.2 if pred.priority == "high" else 0.0)

        return TriageResult(
            customer_id=turn.author_id,
            detected_language="en",
            sentiment_score=sentiment,
            is_urgent=(pred.priority == "high" or has_prof),
            predicted_intent=pred.predicted_intent,
            intent_confidence=pred.confidence,
            extracted_entities={"intent_id": pred.intent_id, "auto_handle": pred.auto_handle_candidate}
        )
