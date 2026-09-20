"""
Abstract Agent Interfaces and Protocol Contracts.
Defines the architectural contracts for the multi-agent conversational support pipeline.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..schemas.models import Turn, ConversationThread

@dataclass
class TriageResult:
    """Output contract of the Triage & Intent Agent."""
    customer_id: str
    detected_language: str
    sentiment_score: float  # -1.0 (very negative) to +1.0 (very positive)
    is_urgent: bool
    predicted_intent: str
    intent_confidence: float
    extracted_entities: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyArticle:
    """Retrieved support documentation or runbook snippet."""
    article_id: str
    title: str
    category: str
    content: str
    canonical_url: str
    relevance_score: float

@dataclass
class PolicyContext:
    """Output contract of the Policy & Knowledge Retrieval Agent."""
    intent: str
    query: str
    relevant_articles: List[PolicyArticle] = field(default_factory=list)
    actionable_resolution_steps: List[str] = field(default_factory=list)
    is_authenticated_action_required: bool = False

@dataclass
class AgentResponse:
    """Output contract of the Resolution & Response Generation Agent."""
    thread_id: str
    response_text: str
    cited_links: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    generation_confidence: float = 0.0

@dataclass
class EscalationDecision:
    """Output contract of the Safety & Escalation Agent."""
    should_escalate: bool
    escalation_reason: Optional[str] = None
    target_queue: Optional[str] = None  # "human_support", "billing_tier2", "trust_and_safety"
    requires_pii_intake: bool = False
    safe_to_publish: bool = True

# ==============================================================================
# Abstract Agent Interfaces
# ==============================================================================

class ITriageAgent(ABC):
    """Responsible for initial ticket intake, sentiment, language, and intent routing."""

    @abstractmethod
    def triage_message(self, turn: Turn) -> TriageResult:
        """Analyzes an incoming customer message and determines intent and urgency."""
        pass

class IPolicyKnowledgeAgent(ABC):
    """Responsible for knowledge base retrieval, FAQ mapping, and policy verification."""

    @abstractmethod
    def retrieve_context(self, query: str, intent: str) -> PolicyContext:
        """Retrieves official policy articles and troubleshooting runbooks."""
        pass

class IResolverAgent(ABC):
    """Responsible for formulating empathetic, policy-grounded support responses."""

    @abstractmethod
    def generate_response(
        self,
        thread: ConversationThread,
        policy_context: PolicyContext
    ) -> AgentResponse:
        """Synthesizes a brand-consistent response conditioned on dialogue state and policy."""
        pass

class IEscalationAgent(ABC):
    """Responsible for safety guardrails, confidence threshold gating, and human handoff."""

    @abstractmethod
    def evaluate_escalation(
        self,
        thread: ConversationThread,
        proposed_response: AgentResponse
    ) -> EscalationDecision:
        """Evaluates whether to publish the response or hand off to human agents."""
        pass
