"""
Pipeline State Machine and Stage State Container.
Maintains state monotonically passing across each stage of the resolver orchestration pipeline.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PipelineStageTrace:
    """Telemetry record for a single stage execution."""
    stage_name: str
    latency_ms: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineState:
    """
    Mutable state object passed across all pipeline stages:
    Intent Triage -> Hybrid Retrieval -> Context Compression -> Policy Retrieval -> Response Synthesis -> Grounding Validation.
    """
    customer_message: str
    stage_name: str = "INITIALIZED"
    intent_prediction: Optional[Any] = None
    retrieved_conversations: List[Any] = field(default_factory=list)
    compressed_context: Optional[Any] = None
    policy_context: Optional[Any] = None
    resolver_response: Optional[Any] = None
    grounding_report: Optional[Any] = None
    stage_traces: List[PipelineStageTrace] = field(default_factory=list)
    provider_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)

    def transition_to(self, new_stage: str):
        """Transitions state to a subsequent pipeline stage."""
        self.stage_name = new_stage

    def record_trace(self, stage_name: str, latency_ms: float, confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Appends a stage latency and confidence record."""
        trace = PipelineStageTrace(
            stage_name=stage_name,
            latency_ms=round(latency_ms, 2),
            confidence=round(confidence, 4),
            metadata=metadata or {}
        )
        self.stage_traces.append(trace)

    def record_error(self, stage_name: str, error: Exception):
        """Records an error encountered during a stage."""
        self.errors.append({
            "stage_name": stage_name,
            "error_type": type(error).__name__,
            "error_message": str(error)
        })

    def get_latency_breakdown(self) -> Dict[str, float]:
        """Calculates per-stage latency and end-to-end total in milliseconds."""
        breakdown = {t.stage_name: t.latency_ms for t in self.stage_traces}
        total_ms = round(sum(breakdown.values()), 2)
        breakdown["total"] = total_ms
        return breakdown
