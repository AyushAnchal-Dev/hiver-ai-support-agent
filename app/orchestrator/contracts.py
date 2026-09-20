"""
Resolver Orchestrator Contracts and Output Models.
Defines the final ResolverResult contract returned by the orchestration pipeline.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from app.orchestrator.state import PipelineStageTrace

@dataclass
class ResolverResult:
    """Comprehensive output contract returned by the ResolverOrchestrator."""
    customer_message: str
    intent: str
    confidence: float
    response: str
    citations: List[str]
    retrieved_evidence: List[Dict[str, Any]]
    policy_ids: List[str]
    context_summary: str
    matched_articles: List[Dict[str, Any]]
    grounding_score: float
    grounding_report: Optional[Any] = None
    escalation_metadata: Dict[str, Any] = field(default_factory=dict)
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    trace: List[PipelineStageTrace] = field(default_factory=list)
    resolution_type: str = "SELF_SERVE"
    provider_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes ResolverResult to a JSON-compatible dictionary."""
        d = {
            "customer_message": self.customer_message,
            "intent": self.intent,
            "confidence": self.confidence,
            "response": self.response,
            "citations": self.citations,
            "retrieved_evidence": self.retrieved_evidence,
            "policy_ids": self.policy_ids,
            "context_summary": self.context_summary,
            "matched_articles": self.matched_articles,
            "grounding_score": self.grounding_score,
            "escalation_metadata": self.escalation_metadata,
            "latency_breakdown": self.latency_breakdown,
            "trace": [
                {
                    "stage_name": t.stage_name,
                    "latency_ms": t.latency_ms,
                    "confidence": t.confidence,
                    "metadata": t.metadata
                }
                for t in self.trace
            ],
            "resolution_type": str(self.resolution_type),
            "provider_metadata": self.provider_metadata
        }
        if self.grounding_report is not None and hasattr(self.grounding_report, "__dict__"):
            d["grounding_report"] = {
                "is_grounded": getattr(self.grounding_report, "is_grounded", True),
                "final_grounding_score": getattr(self.grounding_report, "final_grounding_score", 1.0),
                "remediation_applied": getattr(self.grounding_report, "remediation_applied", False),
                "unsupported_claims": getattr(self.grounding_report, "unsupported_claims", []),
                "verified_claims": getattr(self.grounding_report, "verified_claims", [])
            }
        return d
