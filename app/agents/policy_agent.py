"""
Deterministic Policy & Knowledge Retrieval Agent.
Maps customer intent and query tokens to canonical Spotify policies, help articles,
troubleshooting SOPs, and escalation triggers without LLM or vector search.
"""
import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PolicyContext:
    """Structured policy guidance retrieved from the deterministic knowledge base."""
    intent: str
    matched_policies: List[Dict[str, Any]]
    matched_articles: List[Dict[str, Any]]
    actionable_steps: List[str]
    canonical_urls: List[str]
    policy_ids: List[str]
    escalation_required: bool
    target_queue: Optional[str]
    escalation_reason: Optional[str]
    dm_required: bool
    subintent: Optional[str] = None

def classify_access_subintent(text: str) -> str:
    """Classifies granular sub-intents for ACCOUNT_ACCESS_AUTH inquiries."""
    t_lower = (text or "").lower()
    if any(w in t_lower for w in ["hacked", "someone logged in", "unauthorized", "changed email", "stolen", "someone's using", "stranger"]):
        return "ACCOUNT_COMPROMISE"
    if any(w in t_lower for w in ["abroad", "14 days", "country restriction", "travelling", "traveling", "different country", "kenya", "trip"]):
        return "TRAVEL_RESTRICTION"
    if any(w in t_lower for w in ["forgot password", "reset password", "reset email", "change password", "new password", "password reset"]):
        return "PASSWORD_RESET"
    if any(w in t_lower for w in ["login unavailable", "page not found", "login failed", "cannot login", "can't login", "unable to login", "can't access", "won't load", "not logged in", "locked out", "can't use account", "login"]):
        return "LOGIN_FAILED"
    return "UNKNOWN_ACCESS"

class PolicyAgent:
    """Deterministic knowledge retrieval engine querying local policy files."""

    def __init__(self, knowledge_dir: Optional[str] = None):
        if knowledge_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            self.knowledge_dir = root_dir / "knowledge"
        else:
            self.knowledge_dir = Path(knowledge_dir)

        # Load knowledge base files
        self.help_articles = self._load_json("spotify_help_articles.json", default=[])
        self.billing_policy = self._load_json("billing_policy.json", default={})
        self.playback_policy = self._load_json("playback_policy.json", default={})
        self.troubleshooting_policy = self._load_json("troubleshooting_policy.json", default={})
        self.escalation_policy = self._load_json("escalation_policy.json", default={})

    def _load_json(self, filename: str, default: Any) -> Any:
        path = self.knowledge_dir / filename
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def evaluate_policy(
        self,
        query: str,
        predicted_intent: str,
        priority: str = "Medium"
    ) -> PolicyContext:
        """
        Deterministically evaluates query and intent against knowledge base.
        Returns matching runbook steps, policy citations, and escalation requirements.
        """
        q_lower = query.lower()
        matched_policies = []
        matched_articles = []
        actionable_steps = []
        canonical_urls = []
        policy_ids = []

        escalation_required = False
        target_queue = None
        escalation_reason = None
        dm_required = False

        subintent = None
        if predicted_intent == "ACCOUNT_ACCESS_AUTH":
            subintent = classify_access_subintent(query)

        # 1. Match Canonical Help Articles by Intent
        for art in self.help_articles:
            if art.get("intent") == predicted_intent:
                matched_articles.append(art)
                policy_ids.append(art["article_id"])
                if art.get("canonical_url"):
                    canonical_urls.append(art["canonical_url"])
                for step in art.get("diagnostic_steps", []):
                    # Filter out 14-day travel step unless subintent is TRAVEL_RESTRICTION
                    if "14 days" in step.lower() or "abroad" in step.lower():
                        if subintent == "TRAVEL_RESTRICTION":
                            actionable_steps.append(step)
                    else:
                        actionable_steps.append(step)

        # 2. Match Domain Policies
        if predicted_intent == "SUBSCRIPTION_BILLING":
            rules = self.billing_policy.get("rules", [])
            # Rule 1: Post-cancellation, unexpected downgrade, or refund
            if any(w in q_lower for w in ["cancel", "cancelled", "cancelling", "refund", "charged after", "premium to free", "changed from premium", "lost premium", "downgrade", "downgraded"]):
                rule = next((r for r in rules if r["policy_id"] == "POL_BILL_01"), None)
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    escalation_required = True
                    target_queue = rule.get("target_queue", "billing_tier2")
                    escalation_reason = "Billing refund / post-cancellation charge dispute requires private account verification."
                    dm_required = True
                    actionable_steps.append(rule["mandatory_instruction"])
            # Rule 2: Double charge
            elif any(w in q_lower for w in ["twice", "double", "two times", "duplicate", "2 times"]):
                rule = next((r for r in rules if r["policy_id"] == "POL_BILL_02"), None)
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    escalation_required = True
                    target_queue = rule.get("target_queue", "billing_tier2")
                    escalation_reason = "Duplicate billing requires private transaction verification."
                    dm_required = True
                    actionable_steps.append(rule["mandatory_instruction"])
            # Rule 4: Third-party billing
            elif any(w in q_lower for w in ["apple", "itunes", "google play", "carrier"]):
                rule = next((r for r in rules if r["policy_id"] == "POL_BILL_04"), None)
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    actionable_steps.append(rule["mandatory_instruction"])
            else:
                # Default billing inquiry
                rule = next((r for r in rules if r["policy_id"] == "POL_BILL_03"), None)
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    actionable_steps.append(rule["mandatory_instruction"])

        elif predicted_intent == "PLAYBACK_STREAMING":
            rules = self.playback_policy.get("rules", [])
            # Check stutter/pause vs sound output
            if any(w in q_lower for w in ["pause", "pausing", "stutter", "buffer", "buffering", "stops"]):
                rule = next((r for r in rules if r["policy_id"] == "POL_PLAY_01"), None)
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    actionable_steps.extend(rule.get("troubleshooting_sequence", []))
            elif any(w in q_lower for w in ["no sound", "silent", "volume", "can't hear"]):
                rule = next((r for r in rules if r["policy_id"] == "POL_PLAY_02"), None)
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    actionable_steps.extend(rule.get("troubleshooting_sequence", []))
            else:
                rule = rules[0] if rules else None
                if rule:
                    matched_policies.append(rule)
                    policy_ids.append(rule["policy_id"])
                    actionable_steps.extend(rule.get("troubleshooting_sequence", []))

        # 3. Match Troubleshooting SOPs
        sops = self.troubleshooting_policy.get("sops", [])
        for sop in sops:
            if predicted_intent in sop.get("applicable_intents", []):
                policy_ids.append(sop["sop_id"])
                # Extract clean steps
                steps_data = sop.get("steps", [])
                if isinstance(steps_data, list):
                    actionable_steps.extend(steps_data)
                elif isinstance(steps_data, dict):
                    # Pick platform if mentioned, else general
                    if "windows" in q_lower:
                        actionable_steps.extend(steps_data.get("windows", []))
                    elif "mac" in q_lower or "os x" in q_lower:
                        actionable_steps.extend(steps_data.get("mac", []))
                    elif any(w in q_lower for w in ["android", "iphone", "ios", "phone"]):
                        actionable_steps.extend(steps_data.get("mobile", []))
                    else:
                        # Add desktop default
                        actionable_steps.extend(steps_data.get("windows", [])[:3])

        # 4. Global Escalation Triggers Check
        triggers = self.escalation_policy.get("escalation_triggers", [])
        # Check PII requirement or account compromise
        if any(w in q_lower for w in ["hacked", "stolen", "unauthorized", "stranger", "breach"]) or subintent == "ACCOUNT_COMPROMISE":
            trig = next((t for t in triggers if t["trigger_id"] == "ESC_ACCOUNT_COMPROMISE"), None)
            if trig:
                escalation_required = True
                target_queue = trig["target_queue"]
                escalation_reason = "Customer reported potential account compromise."
                dm_required = True
                policy_ids.append(trig["trigger_id"])
        elif any(w in q_lower for w in ["email address", "credit card", "receipt number", "statement"]):
            trig = next((t for t in triggers if t["trigger_id"] == "ESC_PII_SECURITY"), None)
            if trig:
                escalation_required = True
                target_queue = trig["target_queue"]
                escalation_reason = "Customer message contains or requests PII verification."
                dm_required = True
                policy_ids.append(trig["trigger_id"])
        elif any(w in q_lower for w in ["drink my own", "you lied to me", "you hurt me", "deleting the app", "piss", "harm", "depressed", "distress", "suicide"]):
            escalation_required = True
            target_queue = "human_support"
            escalation_reason = "Customer distress / safety grievance requires human support specialist."
            dm_required = True
            policy_ids.append("ESC_SAFETY")

        # Deduplicate policy_ids and actionable_steps while preserving order
        unique_policy_ids = []
        for pid in policy_ids:
            if pid not in unique_policy_ids:
                unique_policy_ids.append(pid)

        unique_steps = []
        for st in actionable_steps:
            if st not in unique_steps:
                unique_steps.append(st)

        return PolicyContext(
            intent=predicted_intent,
            matched_policies=matched_policies,
            matched_articles=matched_articles,
            actionable_steps=unique_steps[:6],
            canonical_urls=canonical_urls[:2],
            policy_ids=unique_policy_ids,
            escalation_required=escalation_required,
            target_queue=target_queue,
            escalation_reason=escalation_reason,
            dm_required=dm_required,
            subintent=subintent
        )

    def retrieve_policy(self, predicted_intent: str, query: str = "", priority: str = "Medium") -> PolicyContext:
        """API compatibility alias supporting retrieve_policy(predicted_intent, query)."""
        return self.evaluate_policy(query=query, predicted_intent=predicted_intent, priority=priority)

    def retrieve_context(self, query: str, intent: str) -> PolicyContext:
        """IPolicyKnowledgeAgent protocol contract compatibility alias."""
        return self.evaluate_policy(query=query, predicted_intent=intent)

