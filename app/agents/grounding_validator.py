"""
Groundedness & Anti-Hallucination Validator.
Verifies every actionable diagnostic step, claim, and URL against retrieved evidence snippets
and canonical policy documents. Intercepts and remediates unsupported claims before publication.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from app.agents.context_builder import CompressedContext, EvidenceSnippet
from app.agents.policy_agent import PolicyContext

@dataclass
class GroundingReport:
    """Audit report produced for every generated support response."""
    is_grounded: bool
    grounding_score: float             # Bounded [0.0, 1.0] (alias for final_grounding_score)
    total_claims: int
    verified_claims: List[str]
    unsupported_claims: List[str]      # Unsupported claims detected before remediation
    remediation_applied: bool
    final_response: str
    raw_grounding_score: float = 1.0   # Bounded [0.0, 1.0] (pre-remediation score)
    final_grounding_score: float = 1.0 # Bounded [0.0, 1.0] (post-remediation score)
    unsupported_claims_before: List[str] = field(default_factory=list)
    unsupported_claims_after: List[str] = field(default_factory=list)
    claim_classifications: List[Dict[str, str]] = field(default_factory=list)
    audit_trace: Dict[str, Any] = field(default_factory=dict)

class GroundingValidator:
    """Validates factual grounding, checks link authenticity, and eliminates hallucinations."""

    # Canonical Spotify domain patterns
    CANONICAL_LINK_PATTERNS = [
        r"https?://support\.spotify\.com\S*",
        r"https?://community\.spotify\.com\S*",
        r"https?://spotify\.com\S*",
        r"https?://spoti\.fi\S*",
        r"<LINK:[^>]+>"
    ]

    # Prohibited hallucination signatures
    PROHIBITED_PATTERNS = [
        (r"\b1-?800-?[0-9\-]+\b", "Unsupported phone support: Spotify does not offer phone customer support."),
        (r"\b(call us at|our phone number is)\b", "Unsupported phone contact method."),
        (r"\b(we have issued a full refund|your account has been credited)\b", "Unauthorized public financial guarantee without private billing verification."),
        (r"\b(registry editor|regedit|bios settings)\b", "Dangerous out-of-domain technical diagnostic advice.")
    ]

    # Brand voice whitelist patterns (empathy acknowledgements, closings, diagnostic inquiries)
    BRAND_VOICE_WHITELIST = [
        r"\b(hey|hi|hello)\b",
        r"\b(thanks for reaching out)\b",
        r"\b(we hear you)\b",
        r"\b(that doesn'?t sound right)\b",
        r"\b(we('?re| are) here to help)\b",
        r"\b(lend a hand)\b",
        r"\b(we('?re| are) always happy to chat music)\b",
        r"\b(we know billing concerns.*frustrating)\b",
        r"\b(payment questions can definitely be stressful)\b",
        r"\b(music that randomly pauses is no fun)\b",
        r"\b(playback glitch doesn'?t sound right)\b",
        r"\b(sorry to hear the app is crashing)\b",
        r"\b(locked out of your account is a hassle)\b",
        r"\b(offline listening should be seamless)\b",
        r"\b(missing music or playlist trouble is alarming)\b",
        r"\b(seamless connection across your speakers)\b",
        r"\b(disappointing when a track or album.*isn'?t available)\b",
        r"\b(thanks for bringing this ad experience to our attention)\b",
        r"\b(thanks so much for sharing your product feedback)\b",
        r"\b(really sorry your experience hasn'?t been great)\b",
        r"\b(traveling abroad can sometimes affect account access)\b",
        r"\b(we take account security very seriously)\b",
        r"\b(resetting your password should be quick and easy)\b",
        r"\b(let'?s get your tunes playing smoothly again)\b",
        r"\b(let'?s troubleshoot this together)\b",
        r"\b(let'?s get things back up and running)\b",
        r"\b(we('?ve)? got steps to get you back in)\b",
        r"\b(let'?s see why your downloads aren'?t cooperating)\b",
        r"\b(we have tools to restore your library)\b",
        r"\b(let'?s reconnect)\b",
        r"\b(let'?s check on your subscription status)\b",
        r"\b(let us know how you get on)\b",
        r"/spotifycares",
        r"\b(cheers)\b",
        r"\b(hope this helps)\b",
        r"\b(just let us know)\b",
        r"\b(could you (give us|let us know|tell us|provide).*details.*(device model|app version|operating system))\b",
        r"\b(could you try these quick diagnostic steps)\b",
        r"\b(what('?s| is) happening)\b",
        r"\b(which device)\b",
        r"\b(what device)\b",
        r"\b(device model and app version)\b",
        r"\b(because billing and account details need to remain private)\b",
        r"\b(please send us a direct message with your account('?s)? email address)\b",
        r"\b(send us a direct message)\b",
        r"\b(reach out via dm)\b",
        r"\b(direct message with your registered email)\b",
        r"\b(to investigate this securely)\b",
        r"\b(spotify community idea submissions board)\b",
        r"\b(submit your feature concept or vote on existing ideas)\b",
    ]

    def __init__(self, min_token_overlap: float = 0.35):
        self.min_token_overlap = min_token_overlap

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """Extracts individual actionable sentences or bullet points for verification."""
        # Strip citation block from claim analysis
        cleaned_text = re.sub(r"\[Citations:.*?\]", "", text, flags=re.IGNORECASE).strip()
        
        # Split by newlines or sentence terminators (. ! ?)
        raw_units = re.split(r"(?<=[.!?])\s+|\n+", cleaned_text)
        claims = []
        for u in raw_units:
            u_clean = u.strip().lstrip("-•*0123456789. ")
            if not u_clean:
                continue
            claims.append(u_clean)
        return claims

    def _is_brand_voice_claim(self, claim: str) -> bool:
        """Determines whether a claim is an approved empathetic courtesy or diagnostic phrasing."""
        claim_low = claim.lower()
        for pat in self.BRAND_VOICE_WHITELIST:
            if re.search(pat, claim_low):
                return True
        # Also check short conversational courtesies
        if any(claim_low.startswith(p) for p in ["hey", "hi", "hello", "thanks", "cheers", "let us know"]) and len(claim.split()) < 12:
            return True
        return False

    def _build_knowledge_token_set(
        self,
        context: CompressedContext,
        policy: PolicyContext
    ) -> Tuple[set, str]:
        """Aggregates all authorized words and phrases from evidence snippets and policy."""
        corpus_text_parts = []

        # 1. From retrieved evidence
        for snip in context.snippets:
            corpus_text_parts.append(snip.content)

        # 2. From policy documents
        for step in policy.actionable_steps:
            corpus_text_parts.append(step)
        for art in policy.matched_articles:
            corpus_text_parts.append(art.get("summary", ""))
            corpus_text_parts.append(art.get("title", ""))
        for pol in policy.matched_policies:
            corpus_text_parts.append(pol.get("description", ""))
            corpus_text_parts.append(pol.get("mandatory_instruction", ""))

        full_reference_text = " ".join(corpus_text_parts).lower()
        # Word set ignoring stop tokens
        tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", full_reference_text))
        return tokens, full_reference_text

    def _classify_claim(
        self,
        claim: str,
        context: CompressedContext,
        policy: PolicyContext,
        knowledge_tokens: set,
        ref_text: str
    ) -> Tuple[str, str]:
        """
        Classifies claim into:
        - brand_voice_claim
        - evidence_claim
        - policy_claim
        - article_claim
        - unsupported_claim
        Returns: (claim_type, source)
        """
        c_low = claim.lower()

        # 1. Check prohibited patterns
        for pat, reason in self.PROHIBITED_PATTERNS:
            if re.search(pat, c_low):
                return "unsupported_claim", "prohibited_signature"

        # 2. Check for invalid links
        links = re.findall(r"https?://\S+|<LINK:[^>]+>", claim)
        for link in links:
            is_valid = any(re.match(pat, link, re.IGNORECASE) for pat in self.CANONICAL_LINK_PATTERNS)
            if not is_valid:
                return "unsupported_claim", "invalid_link"

        # 3. Check Brand Voice Whitelist
        if self._is_brand_voice_claim(claim):
            return "brand_voice_claim", "brand_voice"

        # Meaningful tokens from claim
        c_tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", c_low))
        meaningful = {t for t in c_tokens if t not in {
            "the", "and", "you", "your", "can", "could", "please", "with",
            "for", "this", "that", "from", "have", "are", "will", "our"
        }}

        if not meaningful:
            return "brand_voice_claim", "brand_voice"

        # 4. Check Policy actionable steps & matched policies
        for step in policy.actionable_steps:
            st_tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", step.lower()))
            if len(meaningful & st_tokens) / len(meaningful) >= 0.25:
                pid = policy.policy_ids[0] if policy.policy_ids else "policy_rule"
                return "policy_claim", pid

        for pol in policy.matched_policies:
            pol_text = (pol.get("description", "") + " " + pol.get("mandatory_instruction", "")).lower()
            pol_tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", pol_text))
            if len(meaningful & pol_tokens) / len(meaningful) >= 0.25:
                return "policy_claim", pol.get("policy_id", "policy_rule")

        # 5. Check Help Articles
        for art in policy.matched_articles:
            art_text = (art.get("title", "") + " " + art.get("summary", "")).lower()
            art_tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", art_text))
            if len(meaningful & art_tokens) / len(meaningful) >= 0.25:
                return "article_claim", art.get("article_id", "help_article")

        # 6. Check Retrieved Evidence Snippets
        for snip in context.snippets:
            snip_tokens = set(re.findall(r"\b[a-z0-9_]{3,}\b", snip.content.lower()))
            if len(meaningful & snip_tokens) / len(meaningful) >= 0.25:
                return "evidence_claim", f"Thread #{snip.thread_id}"

        # 7. Check overall knowledge overlap
        overlap = len(meaningful & knowledge_tokens) / len(meaningful)
        if overlap >= self.min_token_overlap:
            return "policy_claim", "knowledge_base"

        # 8. Fallback
        return "unsupported_claim", "none"

    def validate_response(
        self,
        proposed_response: str,
        context: CompressedContext,
        policy: PolicyContext
    ) -> GroundingReport:
        """
        Runs comprehensive factual and link validation on proposed response.
        Classifies every claim, intercepts prohibited patterns, and applies automatic remediation.
        """
        claims = self._extract_claims(proposed_response)
        knowledge_tokens, ref_text = self._build_knowledge_token_set(context, policy)

        verified_claims = []
        unsupported_claims_before = []
        claim_classifications = []
        remediated_text = proposed_response

        # Check 1: Prohibited Hallucination Signatures in full text
        for pattern, reason in self.PROHIBITED_PATTERNS:
            match = re.search(pattern, remediated_text, re.IGNORECASE)
            if match:
                hallucination_str = match.group(0)
                unsupported_claims_before.append(f"Prohibited pattern '{hallucination_str}': {reason}")
                remediated_text = re.sub(pattern, "[REDACTED: Unsupported contact or claim]", remediated_text, flags=re.IGNORECASE)

        # Check 2: Canonical Link Verification in full text
        links_in_response = re.findall(r"https?://\S+|<LINK:[^>]+>", proposed_response)
        for link in links_in_response:
            is_valid_link = any(re.match(pat, link, re.IGNORECASE) for pat in self.CANONICAL_LINK_PATTERNS)
            if not is_valid_link:
                unsupported_claims_before.append(f"Non-canonical link detected: '{link}'")
                remediated_text = remediated_text.replace(link, "<LINK:URL>")

        # Check 3: Detailed Claim-by-Claim Verification
        for claim in claims:
            claim_type, source = self._classify_claim(
                claim=claim,
                context=context,
                policy=policy,
                knowledge_tokens=knowledge_tokens,
                ref_text=ref_text
            )

            claim_classifications.append({
                "claim": claim,
                "claim_type": claim_type,
                "source": source
            })

            if claim_type == "unsupported_claim":
                if claim not in unsupported_claims_before:
                    unsupported_claims_before.append(claim)
                # Remediate by removing from response
                remediated_text = remediated_text.replace(claim, "")
            else:
                verified_claims.append(claim)

        # Clean residual punctuation/spaces from remediation
        remediated_text = re.sub(r"\n\s*\n+", "\n\n", remediated_text).strip()
        remediated_text = re.sub(r"[ \t]+", " ", remediated_text)

        total_claims_count = len(verified_claims) + len(unsupported_claims_before)
        if total_claims_count == 0:
            raw_score = 1.0
        else:
            raw_score = round(len(verified_claims) / total_claims_count, 4)

        # If remediation pruned too much, fallback to official policy step
        if len(remediated_text.split()) < 10 and policy.actionable_steps:
            fallback_step = policy.actionable_steps[0]
            remediated_text = f"Hey there! We're here to help. {fallback_step} /SpotifyCares"
            if policy.policy_ids:
                remediated_text += f" [Citations: Policy {policy.policy_ids[0]}]"

        # Check post-remediation remaining unsupported claims (should be empty)
        unsupported_claims_after = []
        for pattern, _ in self.PROHIBITED_PATTERNS:
            if re.search(pattern, remediated_text, re.IGNORECASE):
                unsupported_claims_after.append("Remaining prohibited pattern")

        final_score = 1.0 if len(unsupported_claims_after) == 0 else 0.5
        is_grounded = (len(unsupported_claims_before) == 0)
        remediation_applied = (len(unsupported_claims_before) > 0)

        return GroundingReport(
            is_grounded=is_grounded,
            grounding_score=final_score,
            total_claims=total_claims_count,
            verified_claims=verified_claims,
            unsupported_claims=unsupported_claims_before,
            remediation_applied=remediation_applied,
            final_response=remediated_text,
            raw_grounding_score=raw_score,
            final_grounding_score=final_score,
            unsupported_claims_before=unsupported_claims_before,
            unsupported_claims_after=unsupported_claims_after,
            claim_classifications=claim_classifications,
            audit_trace={
                "claims_evaluated": total_claims_count,
                "verified_count": len(verified_claims),
                "unsupported_before_count": len(unsupported_claims_before),
                "unsupported_after_count": len(unsupported_claims_after),
                "raw_grounding_score": raw_score,
                "final_grounding_score": final_score,
                "link_count": len(links_in_response)
            }
        )
