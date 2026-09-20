# Hiver AI Support Agent — API Reference

This document provides complete technical specifications for all REST endpoints exposed by the **Hiver AI Support Agent** service.

- **Base URL**: `http://localhost:8000`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **OpenAPI JSON Spec**: `http://localhost:8000/openapi.json`

---

## Table of Contents
1. [POST /resolve](#1-post-resolve)
2. [GET /health](#2-get-health)
3. [GET /ready](#3-get-ready)
4. [GET /](#4-get-)
5. [GET /metrics](#5-get-metrics)
6. [Error Responses & Status Codes](#6-error-responses--status-codes)

---

## 1. POST /resolve

Resolves an incoming customer support inquiry through the complete 6-stage grounded RAG pipeline.

### Request

- **Method**: `POST`
- **Path**: `/resolve`
- **Content-Type**: `application/json`

#### Request Body Schema

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `customer_message` | `string` | **Yes** | — | Raw customer inquiry text (min 2, max 4000 characters). |
| `use_llm` | `boolean` | No | `false` | If `true`, routes synthesis to configured LLM; if `false`, uses deterministic resolver. |
| `priority` | `string` | No | `null` | Optional priority override (`P1`, `P2`, `P3`). Defaults to intent priority. |

#### Example Request
```json
{
  "customer_message": "I cancelled my Premium subscription last week but you took payment from my PayPal today.",
  "use_llm": false,
  "priority": "P2"
}
```

```bash
curl -X POST "http://localhost:8000/resolve" \
     -H "Content-Type: application/json" \
     -d '{"customer_message": "I cancelled my Premium subscription last week but you took payment from my PayPal today."}'
```

---

### Response

- **Status Code**: `200 OK`
- **Content-Type**: `application/json`

#### Response Body Schema (ResolverResult)

| Field | Type | Description |
| :--- | :--- | :--- |
| `customer_message` | `string` | Original customer query passed to the resolver. |
| `intent` | `string` | Classified Spotify support intent category. |
| `confidence` | `float` | Calibrated overall confidence score `[0.0 - 1.0]` (4-factor formula). |
| `response` | `string` | Grounded, policy-compliant response text. |
| `citations` | `list[string]` | Verified knowledge base policy IDs and conversation thread citations. |
| `retrieved_evidence` | `list[object]` | Top-5 retrieved conversation evidence items with similarity scores and summaries. |
| `policy_ids` | `list[string]` | Identifiers of all policy runbooks matched for this intent. |
| `context_summary` | `string` | Condensed distillation of customer issue context. |
| `matched_articles` | `list[object]` | Canonical help center article objects matched to the query. |
| `grounding_score` | `float` | Final post-validation claim grounding score `[0.0 - 1.0]`. |
| `grounding_report` | `object` | Detailed grounding audit report (unsupported claims, hallucination flags, remediation status). |
| `escalation_metadata` | `object` | Escalation status, reason, target queue, and DM deflection flag. |
| `latency_breakdown` | `object` | Millisecond execution time per pipeline stage. |
| `trace` | `list[object]` | Ordered audit trace of all 6 pipeline stages. |
| `resolution_type` | `string` | High-level action category (`DM_DEFLECTION`, `PASSWORD_RESET`, etc.). |
| `provider_metadata` | `object` | Telemetry metadata for the provider engine used. |

#### Example Response
```json
{
  "customer_message": "I cancelled my Premium subscription last week but you took payment from my PayPal today.",
  "intent": "SUBSCRIPTION_BILLING",
  "confidence": 0.9292,
  "response": "Hi there! Payment questions can definitely be stressful, so let's check on your subscription status. Because billing and account details need to remain private, please send us a Direct Message with your account's email address and receipt date so our team can assist. Let us know how you get on! /SpotifyCares [Citations: Thread #T101, Policy POL_BILL_01]",
  "citations": [
    "Thread #T101",
    "Policy POL_BILL_01"
  ],
  "retrieved_evidence": [
    {
      "thread_id": "T101",
      "intent": "SUBSCRIPTION_BILLING",
      "similarity_score": 0.8842,
      "customer_issue_summary": "Unexpected subscription charge after cancellation",
      "resolution_summary": "Escalated to billing team via private direct message",
      "retrieved_turns_count": 2
    }
  ],
  "policy_ids": [
    "POL_BILL_01"
  ],
  "context_summary": "Customer reports cancellation followed by unexpected charge via PayPal.",
  "matched_articles": [
    {
      "article_id": "ART_BILL_01",
      "title": "Subscription Billing and Payment Inquiries",
      "canonical_url": "https://support.spotify.com/article/subscription-billing/"
    }
  ],
  "grounding_score": 1.0,
  "grounding_report": {
    "raw_grounding_score": 0.98,
    "final_grounding_score": 1.0,
    "unsupported_claims_count": 0,
    "remediation_applied": false,
    "prohibited_patterns_detected": [],
    "hallucination_flags": [],
    "validated_citations": ["Thread #T101", "Policy POL_BILL_01"]
  },
  "escalation_metadata": {
    "escalation_required": true,
    "target_queue": "tier_2_billing",
    "escalation_reason": "Payment dispute requiring sensitive account lookup",
    "dm_required": true
  },
  "latency_breakdown": {
    "intent_triage": 0.52,
    "hybrid_retrieval": 18.34,
    "context_compression": 0.42,
    "policy_retrieval": 0.08,
    "response_synthesis": 2.15,
    "grounding_validation": 1.28,
    "total": 22.79
  },
  "trace": [
    {
      "stage_name": "intent_triage",
      "latency_ms": 0.52,
      "confidence": 0.95
    },
    {
      "stage_name": "hybrid_retrieval",
      "latency_ms": 18.34,
      "confidence": 0.8842
    },
    {
      "stage_name": "context_compression",
      "latency_ms": 0.42,
      "confidence": 0.85
    },
    {
      "stage_name": "policy_retrieval",
      "latency_ms": 0.08,
      "confidence": 1.0
    },
    {
      "stage_name": "response_synthesis",
      "latency_ms": 2.15,
      "confidence": 0.9292
    },
    {
      "stage_name": "grounding_validation",
      "latency_ms": 1.28,
      "confidence": 1.0
    }
  ],
  "resolution_type": "DM_DEFLECTION",
  "provider_metadata": {
    "engine": "deterministic_resolver",
    "model": "rule_and_corpus_grounded",
    "brand_signature": "/SpotifyCares"
  }
}
```

---

## 2. GET /health

Liveness probe endpoint. Verifies the FastAPI application is alive and returning responses.

### Request
```bash
curl -X GET "http://localhost:8000/health"
```

### Response
- **Status Code**: `200 OK`
```json
{
  "status": "healthy",
  "uptime_seconds": 142.58,
  "version": "1.0.0",
  "is_warmed": true,
  "provider": "MOCK",
  "timestamp": "2026-09-17T20:15:00.000000+00:00"
}
```

---

## 3. GET /ready

Readiness probe endpoint. Verifies whether FAISS indices, embedding models, and policy caches have finished preloading and the service is ready for traffic.

### Request
```bash
curl -X GET "http://localhost:8000/ready"
```

### Response (Ready)
- **Status Code**: `200 OK`
```json
{
  "ready": true,
  "is_warmed": true,
  "warmup_duration_ms": 33177.30,
  "retriever_index_loaded": true
}
```

### Response (Warming Up)
- **Status Code**: `503 Service Unavailable`
```json
{
  "ready": false,
  "is_warmed": false,
  "warmup_duration_ms": 0.0,
  "retriever_index_loaded": false
}
```

---

## 4. GET /

Service root discovery endpoint.

### Request
```bash
curl -X GET "http://localhost:8000/"
```

### Response
- **Status Code**: `200 OK`
```json
{
  "service": "Hiver AI Support Agent API",
  "status": "online",
  "version": "1.0.0",
  "docs_url": "/docs",
  "health_url": "/health",
  "ready_url": "/ready"
}
```

---

## 5. GET /metrics

Returns a real-time JSON snapshot of aggregated execution metrics, cache hit rates, and latency percentiles.

### Request
```bash
curl -X GET "http://localhost:8000/metrics"
```

### Response
- **Status Code**: `200 OK`
```json
{
  "timestamp": "2026-09-17T20:15:00.000000+00:00",
  "total_requests": 24,
  "successful_requests": 24,
  "success_rate": 1.0,
  "escalations_triggered": 6,
  "escalation_rate": 0.25,
  "mean_grounding_score": 1.0,
  "retriever_cache_hits": 10,
  "bm25_cache_hits": 4,
  "context_cache_hits": 10,
  "cache_hit_rate": 0.35,
  "cold_requests": 1,
  "warm_requests": 23,
  "provider_usage": {
    "deterministic_resolver": 24
  },
  "intent_distribution": {
    "ACCOUNT_ACCESS_AUTH": 8,
    "SUBSCRIPTION_BILLING": 8,
    "AUDIO_PLAYBACK_ISSUES": 8
  },
  "resolution_types": {
    "PASSWORD_RESET": 8,
    "DM_DEFLECTION": 8,
    "TECHNICAL_TROUBLESHOOTING": 8
  }
}
```

---

## 6. Error Responses & Status Codes

| Status Code | Reason | Cause | Remediating Action |
| :--- | :--- | :--- | :--- |
| `400 Bad Request` | Validation Error | `customer_message` is missing or whitespace only | Provide a valid query string with at least 2 characters |
| `422 Unprocessable Entity` | Schema Error | Malformed JSON body or invalid field types | Verify request body matches `ResolveRequest` schema |
| `500 Internal Server Error` | Pipeline Crash | Unexpected runtime error in a stage | Check `logs/orchestrator.log` for execution stack trace |
| `503 Service Unavailable` | Not Ready | Retrieval index or embedding model warmup still running | Wait for `/ready` probe to return `200 OK` |
