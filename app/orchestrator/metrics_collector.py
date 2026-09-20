"""
Runtime Metrics Collector and Structured Logger.
Aggregates pipeline execution statistics and persists:
- metrics/pipeline_metrics.json
- metrics/latency_metrics.json
- logs/orchestrator.log
"""
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.orchestrator.contracts import ResolverResult

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "structured_payload") and isinstance(record.structured_payload, dict):
            return json.dumps(record.structured_payload)

        msg = record.getMessage()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": msg
        }
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            payload.update(record.structured_data)
        return json.dumps(payload)

class MetricsCollector:
    """Collects and exports pipeline runtime and latency metrics."""

    def __init__(self, metrics_dir: Optional[Path] = None, logs_dir: Optional[Path] = None):
        root = Path(__file__).resolve().parent.parent.parent
        self.metrics_dir = metrics_dir or (root / "metrics")
        self.logs_dir = logs_dir or (root / "logs")

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_metrics_file = self.metrics_dir / "pipeline_metrics.json"
        self.latency_metrics_file = self.metrics_dir / "latency_metrics.json"
        self.log_file = self.logs_dir / "orchestrator.log"

        # In-memory history
        self.total_requests = 0
        self.successful_requests = 0
        self.escalations_triggered = 0
        self.cold_requests = 0
        self.warm_requests = 0
        self.retriever_cache_hits = 0
        self.bm25_cache_hits = 0
        self.context_cache_hits = 0
        self.cache_hit_rate = 0.0
        self.provider_usage: Dict[str, int] = {}
        self.grounding_scores: List[float] = []
        self.intent_distribution: Dict[str, int] = {}
        self.resolution_types: Dict[str, int] = {}
        self.stage_latencies: Dict[str, List[float]] = {}
        self.total_latencies: List[float] = []

        # Setup dedicated logger with JSON formatter
        self.logger = logging.getLogger("ResolverOrchestrator")
        self.logger.setLevel(logging.INFO)
        for h in list(self.logger.handlers):
            h.close()
        self.logger.handlers.clear()
        handler = logging.FileHandler(str(self.log_file), encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Computes percentile with linear interpolation."""
        if not values:
            return 0.0
        if len(values) == 1:
            return round(float(values[0]), 2)
        sorted_vals = sorted(values)
        idx = (len(sorted_vals) - 1) * p
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        weight = idx - lower
        val = sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight
        return round(float(val), 2)

    def record_execution(
        self,
        result: ResolverResult,
        error: Optional[Exception] = None,
        is_cold: Optional[bool] = None,
        cache_hit_rate: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        retriever_cache_hits: Optional[int] = None,
        bm25_cache_hits: Optional[int] = None,
        context_cache_hits: Optional[int] = None
    ):
        """Records telemetry for an executed pipeline request."""
        self.total_requests += 1
        request_id = str(uuid.uuid4())

        # Track cold vs warm requests
        if is_cold is not None:
            cold_start = bool(is_cold)
            if is_cold:
                self.cold_requests += 1
            else:
                self.warm_requests += 1
        else:
            cold_start = bool(self.total_requests == 1)
            if self.total_requests == 1:
                self.cold_requests += 1
            else:
                self.warm_requests += 1

        if cache_hit_rate is not None:
            self.cache_hit_rate = round(float(cache_hit_rate), 4)

        # Track cache hits from singletons or explicit arguments
        try:
            from app.retriever.semantic_search import SemanticSearch
            self.retriever_cache_hits = SemanticSearch.cache_hits
        except Exception:
            pass

        try:
            from app.retriever.lexical_search import BM25LexicalSearch
            self.bm25_cache_hits = BM25LexicalSearch.cache_hits
        except Exception:
            pass

        if retriever_cache_hits is not None:
            self.retriever_cache_hits = retriever_cache_hits
        if bm25_cache_hits is not None:
            self.bm25_cache_hits = bm25_cache_hits
        if context_cache_hits is not None:
            self.context_cache_hits = context_cache_hits

        hit = bool(cache_hit if cache_hit is not None else (self.cache_hit_rate > 0 or self.context_cache_hits > 0))

        if error:
            error_payload = {
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "intent": getattr(result, "intent", "UNKNOWN"),
                "confidence": 0.0,
                "provider": getattr(result, "provider_metadata", {}).get("engine", "unknown"),
                "grounding_score": 0.0,
                "cache_hit": False,
                "cold_start": cold_start,
                "latency_ms": 0.0,
                "resolution_type": "ERROR",
                "trace": [],
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
            record = self.logger.makeRecord(
                self.logger.name,
                logging.ERROR,
                fn="",
                lno=0,
                msg=f"Execution failed: {type(error).__name__}: {error}",
                args=(),
                exc_info=None
            )
            record.structured_payload = error_payload
            self.logger.handle(record)
            self.flush()
            return

        self.successful_requests += 1
        intent = str(result.intent)
        self.intent_distribution[intent] = self.intent_distribution.get(intent, 0) + 1

        res_type = str(result.resolution_type)
        self.resolution_types[res_type] = self.resolution_types.get(res_type, 0) + 1

        provider = str(result.provider_metadata.get("engine", "unknown"))
        self.provider_usage[provider] = self.provider_usage.get(provider, 0) + 1

        if result.escalation_metadata.get("escalation_required"):
            self.escalations_triggered += 1

        self.grounding_scores.append(result.grounding_score)

        # Stage latencies
        trace_data = []
        for trace_entry in result.trace:
            s_name = trace_entry.stage_name
            if s_name not in self.stage_latencies:
                self.stage_latencies[s_name] = []
            self.stage_latencies[s_name].append(trace_entry.latency_ms)
            if hasattr(trace_entry, "to_dict"):
                trace_data.append(trace_entry.to_dict())
            else:
                trace_data.append({
                    "stage_name": s_name,
                    "latency_ms": trace_entry.latency_ms,
                    "confidence": trace_entry.confidence
                })

        total_ms = result.latency_breakdown.get("total", 0.0)
        self.total_latencies.append(total_ms)

        # Log structured JSON line matching required schema
        structured_payload = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "intent": result.intent,
            "confidence": round(float(result.confidence), 4),
            "provider": provider,
            "grounding_score": round(float(result.grounding_score), 4),
            "cache_hit": hit,
            "cold_start": cold_start,
            "latency_ms": round(float(total_ms), 2),
            "resolution_type": res_type,
            "summary": f"Intent: {result.intent} | Total Latency: {round(float(total_ms), 2)}ms",
            "trace": trace_data
        }

        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            fn="",
            lno=0,
            msg=f"Query resolved for intent {result.intent}",
            args=(),
            exc_info=None
        )
        record.structured_payload = structured_payload
        self.logger.handle(record)

        self.flush()

    def flush(self):
        """Persists metrics data files to disk."""
        # 1. Pipeline Metrics
        mean_grounding = round(sum(self.grounding_scores) / len(self.grounding_scores), 4) if self.grounding_scores else 1.0
        escalation_rate = round(self.escalations_triggered / max(1, self.total_requests), 4)
        success_rate = round(self.successful_requests / max(1, self.total_requests), 4)

        pipeline_metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": success_rate,
            "escalations_triggered": self.escalations_triggered,
            "escalation_rate": escalation_rate,
            "mean_grounding_score": mean_grounding,
            "retriever_cache_hits": self.retriever_cache_hits,
            "bm25_cache_hits": self.bm25_cache_hits,
            "context_cache_hits": self.context_cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "cold_requests": self.cold_requests,
            "warm_requests": self.warm_requests,
            "provider_usage": {str(k): v for k, v in self.provider_usage.items()},
            "intent_distribution": {str(k): v for k, v in self.intent_distribution.items()},
            "resolution_types": {str(k): v for k, v in self.resolution_types.items()},
        }

        with open(self.pipeline_metrics_file, "w", encoding="utf-8") as f:
            json.dump(pipeline_metrics, f, indent=2)

        # 2. Latency Metrics
        latency_stats = {}
        for s_name, l_list in self.stage_latencies.items():
            if l_list:
                latency_stats[s_name] = {
                    "count": len(l_list),
                    "mean_ms": round(sum(l_list) / len(l_list), 2),
                    "min_ms": round(min(l_list), 2),
                    "p50_ms": self._percentile(l_list, 0.50),
                    "p90_ms": self._percentile(l_list, 0.90),
                    "p95_ms": self._percentile(l_list, 0.95),
                    "p99_ms": self._percentile(l_list, 0.99),
                    "max_ms": round(max(l_list), 2)
                }

        if self.total_latencies:
            latency_stats["end_to_end_total"] = {
                "count": len(self.total_latencies),
                "mean_ms": round(sum(self.total_latencies) / len(self.total_latencies), 2),
                "min_ms": round(min(self.total_latencies), 2),
                "p50_ms": self._percentile(self.total_latencies, 0.50),
                "p90_ms": self._percentile(self.total_latencies, 0.90),
                "p95_ms": self._percentile(self.total_latencies, 0.95),
                "p99_ms": self._percentile(self.total_latencies, 0.99),
                "max_ms": round(max(self.total_latencies), 2)
            }

        latency_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_size": len(self.total_latencies),
            "stages": latency_stats
        }

        with open(self.latency_metrics_file, "w", encoding="utf-8") as f:
            json.dump(latency_payload, f, indent=2)

