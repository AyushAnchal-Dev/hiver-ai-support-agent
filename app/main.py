"""
FastAPI Production Application Entry Point.
Exposes the ResolverOrchestrator pipeline via REST endpoints with full OpenAPI documentation,
liveness and readiness probes, and telemetry integration.
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure root path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.settings import get_settings, validate_environment

from app.orchestrator.resolver_orchestrator import ResolverOrchestrator
from app.prompts.prompt_builder import PromptBuilder
from app.schemas.api import ResolveRequest, ResolveResponse, HealthResponse, ReadyResponse

_APP_START_TIME = time.time()
_ORCHESTRATOR: Optional[ResolverOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes and warms up subsystems upon application start."""
    global _ORCHESTRATOR, _APP_START_TIME
    _APP_START_TIME = time.time()

    # 1. Validate environment and required directories
    settings = validate_environment()

    # 2. Ensure prompt template manifest exists
    try:
        pb = PromptBuilder(version=settings.prompt_version)
        pb.save_manifest()
    except Exception as e:
        print(f"[WARN] Prompt manifest verification: {e}")

    # 3. Initialize Resolver Orchestrator
    _ORCHESTRATOR = ResolverOrchestrator(
        settings=settings,
        lazy_init_retriever=False
    )

    # 4. Optional Preload / Warmup
    enable_warmup = os.getenv("ENABLE_WARMUP", "true").lower() in ("true", "1", "yes")
    if enable_warmup:
        try:
            print("[INFO] Warming up ResolverOrchestrator subsystems (embeddings, indexes, policies)...")
            _ORCHESTRATOR.warmup()
            print(f"[INFO] Warmup completed in {_ORCHESTRATOR.warmup_duration_ms:.2f} ms")
        except Exception as e:
            print(f"[WARN] Subsystem warmup error: {e}")

    yield

    # Teardown logic
    if _ORCHESTRATOR and hasattr(_ORCHESTRATOR, "metrics_collector"):
        _ORCHESTRATOR.metrics_collector.flush()

app = FastAPI(
    title="Hiver AI Support Agent API",
    description=(
        "Production Multi-Agent Grounded RAG Support Orchestration Service. "
        "Engineered for customer service automation with 100% policy compliance, "
        "dynamic intent routing, hybrid retrieval (FAISS + BM25), and zero ungrounded hallucinations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "Resolution", "description": "Customer inquiry resolution through the 6-stage grounded RAG pipeline."},
        {"name": "System & Health", "description": "Liveness probes, readiness checks, and runtime telemetry."}
    ],
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def get_orchestrator() -> ResolverOrchestrator:
    """Dependency helper to obtain the initialized orchestrator."""
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = ResolverOrchestrator(lazy_init_retriever=False)
    return _ORCHESTRATOR

@app.get(
    "/",
    tags=["System & Health"],
    summary="API Root Information",
    description="Returns service name, status, documentation URLs, and current active version."
)
async def root():
    return {
        "service": "Hiver AI Support Agent API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "ready_url": "/ready"
    }

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System & Health"],
    summary="Liveness Probe",
    description="Returns operational liveness status, total system uptime, and active LLM provider configuration."
)
async def health_check():
    orch = get_orchestrator()
    settings = get_settings()
    uptime = round(time.time() - _APP_START_TIME, 2)
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        version="1.0.0",
        is_warmed=orch.is_warmed,
        provider=settings.llm_provider,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["System & Health"],
    summary="Readiness Probe",
    description="Indicates whether the dense vector index and retrieval subsystems have completed initial warmup."
)
async def readiness_check():
    orch = get_orchestrator()
    is_ready = orch.is_warmed
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "is_warmed": False,
                "warmup_duration_ms": orch.warmup_duration_ms,
                "retriever_index_loaded": getattr(orch.retriever_agent, "index_loaded", False)
            }
        )

    return ReadyResponse(
        ready=True,
        is_warmed=True,
        warmup_duration_ms=orch.warmup_duration_ms,
        retriever_index_loaded=getattr(orch.retriever_agent, "index_loaded", True)
    )

@app.post(
    "/resolve",
    response_model=ResolveResponse,
    status_code=status.HTTP_200_OK,
    tags=["Resolution"],
    summary="Resolve Customer Inquiry",
    description=(
        "Executes the full 6-stage grounded support pipeline: "
        "Intent Triage -> Hybrid Retrieval (FAISS + BM25) -> Evidence Compression -> "
        "Policy Retrieval -> Response Synthesis -> Grounding Validation."
    ),
    responses={
        200: {
            "description": "Successfully resolved customer inquiry with fully grounded response and citations.",
            "model": ResolveResponse
        },
        400: {"description": "Invalid input query text."},
        500: {"description": "Internal server processing error."}
    }
)
async def resolve_inquiry(request: ResolveRequest):
    if not request.customer_message or not request.customer_message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'customer_message' cannot be blank or empty."
        )

    orch = get_orchestrator()
    try:
        result = orch.resolve(
            customer_message=request.customer_message.strip(),
            use_llm=request.use_llm,
            priority=request.priority
        )

        trace_dicts = [
            t.to_dict() if hasattr(t, "to_dict") else vars(t)
            for t in result.trace
        ]
        grounding_dict = (
            result.grounding_report.to_dict()
            if hasattr(result.grounding_report, "to_dict")
            else vars(result.grounding_report)
        )

        return ResolveResponse(
            customer_message=result.customer_message,
            intent=result.intent,
            confidence=result.confidence,
            response=result.response,
            citations=result.citations,
            retrieved_evidence=result.retrieved_evidence,
            policy_ids=result.policy_ids,
            context_summary=result.context_summary,
            matched_articles=result.matched_articles,
            grounding_score=result.grounding_score,
            grounding_report=grounding_dict,
            escalation_metadata=result.escalation_metadata,
            latency_breakdown=result.latency_breakdown,
            trace=trace_dicts,
            resolution_type=result.resolution_type,
            provider_metadata=result.provider_metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resolver pipeline error: {str(e)}"
        )

@app.get(
    "/metrics",
    tags=["System & Health"],
    summary="Runtime Metrics Snapshot",
    description="Returns aggregated runtime execution counts, latencies, and cache statistics."
)
async def get_metrics():
    orch = get_orchestrator()
    metrics_path = orch.metrics_collector.pipeline_metrics_file
    if metrics_path.exists():
        import json
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "no_metrics_collected"}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)

