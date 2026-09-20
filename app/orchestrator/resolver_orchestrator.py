"""
Production Resolver Orchestration Engine.
Coordinates the end-to-end multi-agent support pipeline:
Customer Query
  -> Stage 1: IntentAgent (Intent Triage)
  -> Stage 2: RetrieverAgent (Hybrid Retrieval)
  -> Stage 3: ContextBuilder (Context Compression)
  -> Stage 4: PolicyAgent (Policy & Knowledge Retrieval)
  -> Stage 5: ResolverAgent / LLM Provider (Response Synthesis)
  -> Stage 6: GroundingValidator (Anti-Hallucination Guardrails)
  -> ResolverResult
"""
import time
from typing import Optional, Dict, Any, List

from app.settings import Settings, get_settings
from app.agents.intent_agent import IntentAgent, IntentPrediction
from app.agents.retriever_agent import RetrieverAgent, RetrievedConversation
from app.agents.context_builder import ContextBuilder, CompressedContext
from app.agents.policy_agent import PolicyAgent, PolicyContext
from app.agents.resolver_agent import ResolverAgent, ResolverResponse, ResponseType
from app.agents.grounding_validator import GroundingValidator, GroundingReport

from app.orchestrator.context_builder import OptimizedContextBuilder
from app.llm.base_provider import BaseLLMProvider
from app.llm.factory import get_llm_provider
from app.prompts.prompt_builder import PromptBuilder
from app.orchestrator.state import PipelineState
from app.orchestrator.contracts import ResolverResult
from app.orchestrator.metrics_collector import MetricsCollector

INTENT_TO_CANONICAL_ARTICLE = {
    "ACCOUNT_ACCESS": "ART_ACC_01",
    "ACCOUNT_ACCESS_AUTH": "ART_ACC_01",
    "SUBSCRIPTION_BILLING": "ART_BILL_01",
    "AUDIO_PLAYBACK_ISSUES": "ART_BUG_01",
    "PLAYLIST_MANAGEMENT": "ART_PLAY_01",
    "OFFLINE_SYNC": "ART_SYNC_01",
    "APP_CRASH_DEVICE_ERROR": "ART_BUG_01",
    "UNKNOWN_OTHER": "ART_BUG_01",
}

class ResolverOrchestrator:
    """Production orchestrator driving the 6-stage grounded support pipeline."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        intent_agent: Optional[IntentAgent] = None,
        retriever_agent: Optional[RetrieverAgent] = None,
        context_builder: Optional[ContextBuilder] = None,
        policy_agent: Optional[PolicyAgent] = None,
        resolver_agent: Optional[ResolverAgent] = None,
        grounding_validator: Optional[GroundingValidator] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        lazy_init_retriever: bool = True
    ):
        self.settings = settings or get_settings()
        self.intent_agent = intent_agent or IntentAgent()
        self.retriever_agent = retriever_agent or RetrieverAgent()
        self.context_builder = context_builder or OptimizedContextBuilder()
        self.policy_agent = policy_agent or PolicyAgent()
        self.resolver_agent = resolver_agent or ResolverAgent(brand_signature=self.settings.brand_signature)
        self.grounding_validator = grounding_validator or GroundingValidator()
        self.llm_provider = llm_provider or get_llm_provider(self.settings)
        self.prompt_builder = prompt_builder or PromptBuilder(version=self.settings.prompt_version)
        self.metrics_collector = metrics_collector or MetricsCollector(
            metrics_dir=self.settings.metrics_dir,
            logs_dir=self.settings.logs_dir
        )
        self.is_degraded: bool = False
        self._is_warm: bool = False
        self.is_warmed: bool = False
        self.warmup_duration_ms: float = 0.0

        if not lazy_init_retriever and hasattr(self.retriever_agent, "_initialized") and not self.retriever_agent._initialized:
            try:
                self.retriever_agent.initialize()
            except Exception as e:
                self.is_degraded = True
                self.metrics_collector.logger.warning(
                    f"Retriever eager initialization failed: {type(e).__name__}: {e}. Orchestrator running in degraded mode."
                )



    def resolve(
        self,
        customer_message: str,
        use_llm: bool = False,
        priority: str = "Medium"
    ) -> ResolverResult:
        """
        Executes the 6-stage orchestration pipeline for an incoming customer message.
        Passes a monotonic PipelineState across all stages and collects latency telemetry.
        """
        state = PipelineState(customer_message=customer_message)

        try:
            # ------------------------------------------------------------------
            # Stage 1: Intent Triage
            # ------------------------------------------------------------------
            state.transition_to("intent_triage")
            t0 = time.perf_counter()
            intent_pred = self.intent_agent.predict(customer_message)
            t_stage1 = (time.perf_counter() - t0) * 1000.0
            state.intent_prediction = intent_pred
            state.record_trace(
                stage_name="intent_triage",
                latency_ms=t_stage1,
                confidence=intent_pred.confidence,
                metadata={"priority": intent_pred.priority, "is_ambiguous": intent_pred.is_ambiguous}
            )

            # ------------------------------------------------------------------
            # Stage 2: Hybrid Retrieval
            # ------------------------------------------------------------------
            state.transition_to("hybrid_retrieval")
            t0 = time.perf_counter()
            retrieved_convs: List[RetrievedConversation] = []
            try:
                retrieved_convs = self.retriever_agent.retrieve(customer_message, top_k=5)
            except Exception as e:
                state.record_error("hybrid_retrieval", e)
                self.is_degraded = True
                self.metrics_collector.logger.warning(
                    f"Hybrid retrieval stage failed: {type(e).__name__}: {e}. Proceeding with empty context."
                )
            t_stage2 = (time.perf_counter() - t0) * 1000.0
            state.retrieved_conversations = retrieved_convs

            mean_sim = (
                sum(c.similarity_score for c in retrieved_convs) / len(retrieved_convs)
                if retrieved_convs else 0.0
            )
            state.record_trace(
                stage_name="hybrid_retrieval",
                latency_ms=t_stage2,
                confidence=mean_sim,
                metadata={"candidates_retrieved": len(retrieved_convs)}
            )

            # ------------------------------------------------------------------
            # Stage 3: Context Compression
            # ------------------------------------------------------------------
            state.transition_to("context_compression")
            t0 = time.perf_counter()
            compressed_ctx = self.context_builder.build_context(retrieved_convs)
            t_stage3 = (time.perf_counter() - t0) * 1000.0
            state.compressed_context = compressed_ctx
            state.record_trace(
                stage_name="context_compression",
                latency_ms=t_stage3,
                confidence=compressed_ctx.mean_retrieval_confidence,
                metadata={
                    "snippet_count": len(compressed_ctx.snippets),
                    "total_chars": compressed_ctx.total_chars,
                    "estimated_tokens": compressed_ctx.estimated_tokens
                }
            )

            # ------------------------------------------------------------------
            # Stage 4: Policy Knowledge Retrieval
            # ------------------------------------------------------------------
            state.transition_to("policy_retrieval")
            t0 = time.perf_counter()
            if hasattr(self.policy_agent, "evaluate_policy"):
                policy_ctx = self.policy_agent.evaluate_policy(
                    query=customer_message,
                    predicted_intent=intent_pred.predicted_intent,
                    priority=priority
                )
            elif hasattr(self.policy_agent, "retrieve_policy"):
                policy_ctx = self.policy_agent.retrieve_policy(
                    intent_pred.predicted_intent,
                    query=customer_message
                )
            else:
                policy_ctx = self.policy_agent.retrieve_context(
                    query=customer_message,
                    intent=intent_pred.predicted_intent
                )
            t_stage4 = (time.perf_counter() - t0) * 1000.0
            state.policy_context = policy_ctx
            policy_conf = 1.0 if policy_ctx.policy_ids else 0.5
            state.record_trace(
                stage_name="policy_retrieval",
                latency_ms=t_stage4,
                confidence=policy_conf,
                metadata={
                    "policy_ids": policy_ctx.policy_ids,
                    "escalation_required": policy_ctx.escalation_required,
                    "dm_required": policy_ctx.dm_required
                }
            )

            # ------------------------------------------------------------------
            # Stage 5: Response Synthesis
            # ------------------------------------------------------------------
            state.transition_to("response_synthesis")
            t0 = time.perf_counter()

            if not use_llm:
                # Deterministic Stage 5 Resolver Engine
                resolver_resp = self.resolver_agent.generate_response(
                    customer_message=customer_message,
                    predicted_intent=intent_pred.predicted_intent,
                    context=compressed_ctx,
                    policy_context=policy_ctx,
                    priority=priority,
                    intent_confidence=intent_pred.confidence
                )
                state.resolver_response = resolver_resp
                proposed_text = resolver_resp.response
                resolution_type = resolver_resp.resolution_type
                citations = resolver_resp.citations
                synthesis_conf = resolver_resp.confidence
                provider_metadata = {
                    "engine": "deterministic_resolver",
                    "model": "rule_and_corpus_grounded",
                    "brand_signature": self.settings.brand_signature
                }
            else:
                # LLM-Assisted Synthesis
                sys_prompt = self.prompt_builder.build_system_prompt()
                user_prompt = self.prompt_builder.build_resolver_prompt(
                    customer_message=customer_message,
                    intent=intent_pred.predicted_intent,
                    context=compressed_ctx,
                    policy=policy_ctx
                )
                llm_gen = self.llm_provider.generate(user_prompt, system_prompt=sys_prompt)
                proposed_text = llm_gen.text

                # Assemble citations
                citations = []
                for tid in compressed_ctx.cited_thread_ids[:2]:
                    citations.append(f"Thread #{tid}")
                for pid in policy_ctx.policy_ids[:2]:
                    citations.append(f"Policy {pid}")

                # Determine resolution type
                resolution_type = (
                    ResponseType("DM_DEFLECTION")
                    if policy_ctx.escalation_required and policy_ctx.dm_required
                    else ResponseType("SELF_SERVE")
                )
                synthesis_conf = 0.90
                provider_metadata = {
                    "engine": self.llm_provider.provider_name,
                    "model": llm_gen.model,
                    "prompt_tokens": llm_gen.prompt_tokens,
                    "completion_tokens": llm_gen.completion_tokens,
                    "total_tokens": llm_gen.total_tokens,
                    "is_fallback": llm_gen.is_fallback,
                    "llm_latency_ms": llm_gen.latency_ms
                }

            t_stage5 = (time.perf_counter() - t0) * 1000.0
            state.provider_metadata = provider_metadata
            state.record_trace(
                stage_name="response_synthesis",
                latency_ms=t_stage5,
                confidence=synthesis_conf,
                metadata={"use_llm": use_llm, "resolution_type": str(resolution_type)}
            )

            # ------------------------------------------------------------------
            # Stage 6: Grounding Validation & Hallucination Guardrails
            # ------------------------------------------------------------------
            state.transition_to("grounding_validation")
            t0 = time.perf_counter()
            grounding_report = self.grounding_validator.validate_response(
                proposed_response=proposed_text,
                context=compressed_ctx,
                policy=policy_ctx
            )
            t_stage6 = (time.perf_counter() - t0) * 1000.0
            state.grounding_report = grounding_report
            state.record_trace(
                stage_name="grounding_validation",
                latency_ms=t_stage6,
                confidence=grounding_report.final_grounding_score,
                metadata={
                    "is_grounded": grounding_report.is_grounded,
                    "remediation_applied": grounding_report.remediation_applied,
                    "unsupported_claims_count": len(grounding_report.unsupported_claims)
                }
            )

            # Assemble Context Summary
            context_summary = (
                f"{len(compressed_ctx.snippets)} snippets from {len(compressed_ctx.cited_thread_ids)} threads "
                f"({compressed_ctx.total_chars} chars, ~{compressed_ctx.estimated_tokens} tokens)"
            )

            # Retrieved Evidence Serialization
            retrieved_evidence_list = []
            for conv in retrieved_convs:
                retrieved_evidence_list.append({
                    "thread_id": conv.thread_id,
                    "intent": conv.intent,
                    "similarity_score": round(conv.similarity_score, 4),
                    "customer_issue_summary": conv.customer_issue_summary,
                    "resolution_summary": conv.resolution_summary,
                    "retrieved_turns_count": len(conv.retrieved_turns)
                })

            # Escalation Metadata
            escalation_metadata = {
                "escalation_required": policy_ctx.escalation_required,
                "target_queue": policy_ctx.target_queue,
                "escalation_reason": policy_ctx.escalation_reason,
                "dm_required": policy_ctx.dm_required
            }

            # Response Confidence Calibration (FIX 8 4-factor formula)
            mean_retrieval_conf = compressed_ctx.mean_retrieval_confidence if compressed_ctx.mean_retrieval_confidence > 0 else 0.5
            intent_conf = min(1.0, max(0.0, float(intent_pred.confidence)))
            grounding_score = grounding_report.final_grounding_score
            raw_conf = (
                0.30 * mean_retrieval_conf +
                0.25 * intent_conf +
                0.25 * grounding_score +
                0.20 * policy_conf
            )
            final_confidence = round(min(1.0, max(0.0, raw_conf)), 4)

            # Build Final ResolverResult
            result = ResolverResult(
                customer_message=customer_message,
                intent=intent_pred.predicted_intent,
                confidence=final_confidence,
                response=grounding_report.final_response,
                citations=citations,
                retrieved_evidence=retrieved_evidence_list,
                policy_ids=policy_ctx.policy_ids,
                context_summary=context_summary,
                matched_articles=policy_ctx.matched_articles,
                grounding_score=grounding_report.final_grounding_score,
                grounding_report=grounding_report,
                escalation_metadata=escalation_metadata,
                latency_breakdown=state.get_latency_breakdown(),
                trace=state.stage_traces,
                resolution_type=str(resolution_type),
                provider_metadata=provider_metadata
            )

            # Record telemetry
            cache_rate = getattr(self.context_builder, "cache_hit_rate", 0.0)
            context_hits = getattr(self.context_builder, "cache_hits", 0)
            is_cache_hit = bool(context_hits > 0 or cache_rate > 0.0)
            self.metrics_collector.record_execution(
                result,
                is_cold=not self._is_warm,
                cache_hit_rate=cache_rate,
                cache_hit=is_cache_hit,
                context_cache_hits=context_hits
            )
            self._is_warm = True
            return result

        except Exception as e:
            state.record_error(state.stage_name, e)
            pred_intent = getattr(state.intent_prediction, "predicted_intent", "UNKNOWN_OTHER")
            fallback_art = INTENT_TO_CANONICAL_ARTICLE.get(pred_intent, "ART_BUG_01")
            # Create a safe, minimal fallback result grounded in canonical knowledge
            fallback_res = ResolverResult(
                customer_message=customer_message,
                intent=pred_intent,
                confidence=0.0,
                response=f"Hey there! Thanks for reaching out to Spotify support. We are looking into this for you. Let us know how you get on! {self.settings.brand_signature} [Citations: Article {fallback_art}]",
                citations=[f"Article {fallback_art}"],
                retrieved_evidence=[],
                policy_ids=[fallback_art],
                context_summary="Pipeline error fallback",
                matched_articles=[],
                grounding_score=0.0,
                grounding_report=None,
                escalation_metadata={"escalation_required": True, "target_queue": "tier1_general"},
                latency_breakdown=state.get_latency_breakdown(),
                trace=state.stage_traces,
                resolution_type="SELF_SERVE",
                provider_metadata={"error": str(e)}
            )
            self.metrics_collector.record_execution(fallback_res, error=e, is_cold=not self._is_warm)
            self._is_warm = True
            return fallback_res

    def warmup(self, queries: Optional[List[str]] = None) -> None:
        """
        Warms up embedding models, index caches, and memory structures
        to ensure subsequent requests achieve sub-50ms latency.
        Exposes:
            is_warmed: bool
            warmup_duration_ms: float
        """
        t0 = time.perf_counter()
        if hasattr(self.retriever_agent, "initialize"):
            self.retriever_agent.initialize()
        elif hasattr(self.retriever_agent, "semantic_retriever"):
            SemanticSearch.preload_model()

        if hasattr(self.policy_agent, "evaluate_policy"):
            self.policy_agent.evaluate_policy("warmup query", "ACCOUNT_ACCESS_AUTH")

        sample_queries = queries or [
            "How do I reset my forgotten password?",
            "Why was I charged twice for Premium?"
        ]
        for q in sample_queries:
            try:
                self.resolve(q, use_llm=False)
            except Exception as e:
                self.metrics_collector.logger.warning(f"Warmup query failed: {e}")

        self.warmup_duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.is_warmed = True
        self._is_warm = True

    @property
    def is_warm(self) -> bool:
        """Backward compatibility alias for is_warmed."""
        return self.is_warmed

