"""
API Request and Response Pydantic Contracts.
Defines schemas with documentation, field constraints, and OpenAPI examples.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ResolveRequest(BaseModel):
    """Customer query submission schema for pipeline resolution."""
    customer_message: str = Field(
        ...,
        description="The raw customer support inquiry text.",
        min_length=2,
        max_length=4000,
        example="I cancelled my Premium subscription last week but you took payment from my PayPal today."
    )
    use_llm: bool = Field(
        default=False,
        description="If True, routes response synthesis to the configured LLM provider; otherwise uses deterministic Stage 5 synthesizer."
    )
    priority: Optional[str] = Field(
        default=None,
        description="Optional manual priority level ('P1', 'P2', 'P3'). Defaults to intent-inferred priority.",
        example="P2"
    )

class ResolveResponse(BaseModel):
    """Complete structured response from the 6-stage resolver orchestrator."""
    customer_message: str = Field(..., description="Original incoming customer query.")
    intent: str = Field(..., description="Predicted Spotify customer inquiry intent category.", example="SUBSCRIPTION_BILLING")
    confidence: float = Field(..., description="Calibrated final confidence score [0.0 - 1.0].", example=0.9183)
    response: str = Field(..., description="Grounded, policy-compliant resolution message for the customer.")
    citations: List[str] = Field(..., description="Cited knowledge base policy and conversation thread IDs.")
    retrieved_evidence: List[Dict[str, Any]] = Field(..., description="Top-5 retrieved historical evidence items.")
    policy_ids: List[str] = Field(..., description="Applicable knowledge base policy identifiers.", example=["POL_BILL_01"])
    context_summary: str = Field(..., description="Condensed summary of past interactions and issue history.")
    matched_articles: List[Any] = Field(..., description="Canonical help center knowledge base article IDs or objects.", example=["ART_BILL_01"])
    grounding_score: float = Field(..., description="Post-validation grounding verification score [0.0 - 1.0].", example=1.0)
    grounding_report: Dict[str, Any] = Field(..., description="Detailed audit breakdown of claim verification and remediations.")
    escalation_metadata: Dict[str, Any] = Field(..., description="Routing instructions, DM deflection requirements, and escalation targets.")
    latency_breakdown: Dict[str, float] = Field(..., description="Per-stage execution latencies in milliseconds.")
    trace: List[Dict[str, Any]] = Field(..., description="Monotonic pipeline execution trace across all 6 stages.")
    resolution_type: str = Field(..., description="Action category (e.g. DM_DEFLECTION, PASSWORD_RESET, TECHNICAL_TROUBLESHOOTING).", example="DM_DEFLECTION")
    provider_metadata: Dict[str, Any] = Field(..., description="Provider engine, model, and token metrics.")

class HealthResponse(BaseModel):
    """Liveness healthcheck response schema."""
    status: str = Field(..., example="healthy")
    uptime_seconds: float = Field(..., example=120.45)
    version: str = Field(..., example="1.0.0")
    is_warmed: bool = Field(..., example=True)
    provider: str = Field(..., example="MOCK")
    timestamp: str = Field(..., example="2026-09-17T20:15:00Z")

class ReadyResponse(BaseModel):
    """Readiness probe response schema."""
    ready: bool = Field(..., example=True)
    is_warmed: bool = Field(..., example=True)
    warmup_duration_ms: float = Field(..., example=34493.75)
    retriever_index_loaded: bool = Field(..., example=True)
