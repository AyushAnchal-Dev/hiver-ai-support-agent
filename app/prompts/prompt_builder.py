"""
Dynamic Prompt Builder and Template Engine.
Supports versioned prompt directories, variable substitution, and guardrail injection.
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

class PromptBuilder:
    """Loads and compiles versioned prompt templates for LLM generation."""

    def __init__(self, version: str = "v1", template_dir: Optional[Path] = None):
        self.version = version
        base_dir = template_dir or Path(__file__).resolve().parent
        self.version_dir = base_dir / version

        if not self.version_dir.exists() or not self.version_dir.is_dir():
            raise FileNotFoundError(f"Prompt template version directory does not exist: {self.version_dir}")

        self.templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self):
        """Loads all markdown template files in the version directory."""
        required = ["system_prompt.md", "resolver_prompt.md", "citation_prompt.md", "guardrail_prompt.md"]
        for fname in required:
            fpath = self.version_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required prompt template '{fname}' in {self.version_dir}")
            with open(fpath, "r", encoding="utf-8") as f:
                self.templates[fname] = f.read().strip()

    def get_version_manifest(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Computes SHA-256 checksums for all prompt templates in the version directory,
        along with a deterministic composite version checksum.
        """
        target_version = version or self.version
        base_dir = self.version_dir.parent / target_version
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"Version directory not found: {base_dir}")

        manifest_templates: Dict[str, str] = {}
        for p in sorted(base_dir.glob("*.md")):
            with open(p, "rb") as f:
                f_hash = hashlib.sha256(f.read()).hexdigest()
            manifest_templates[p.name] = f_hash

        composite_hasher = hashlib.sha256()
        for fname in sorted(manifest_templates.keys()):
            composite_hasher.update(f"{fname}:{manifest_templates[fname]}\n".encode("utf-8"))
        composite_hash = composite_hasher.hexdigest()

        from datetime import datetime, timezone
        return {
            "version": target_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version_checksum": composite_hash,
            "template_count": len(manifest_templates),
            "templates": manifest_templates
        }

    def save_manifest(self, version: Optional[str] = None, output_path: Optional[Path] = None) -> Path:
        """Saves version manifest to JSON file."""
        target_version = version or self.version
        manifest = self.get_version_manifest(target_version)
        out_file = output_path or (self.version_dir.parent / "version_manifest.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        # Also maintain version folder manifest if default was used
        if output_path is None:
            v_manifest = self.version_dir.parent / target_version / "manifest.json"
            v_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(v_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            # Version folder manifest maintained in app/prompts
            pass
        return out_file

    @staticmethod
    def list_available_versions(base_dir: Optional[Path] = None) -> List[str]:
        """Lists available version folders in prompts/."""
        b_dir = base_dir or Path(__file__).resolve().parent
        versions = []
        for p in b_dir.iterdir():
            if p.is_dir() and (p / "system_prompt.md").exists():
                versions.append(p.name)
        return sorted(versions)


    def build_system_prompt(self) -> str:
        """Assembles a full system prompt incorporating persona, guardrails, and citation rules."""
        sys_body = self.templates.get("system_prompt.md", "")
        guardrails = self.templates.get("guardrail_prompt.md", "")
        citations = self.templates.get("citation_prompt.md", "")

        return f"{sys_body}\n\n---\n\n{guardrails}\n\n---\n\n{citations}"

    def build_resolver_prompt(
        self,
        customer_message: str,
        intent: str,
        context: Any,
        policy: Any
    ) -> str:
        """
        Populates resolver_prompt.md with customer inquiry, compressed evidence snippets,
        and official policy runbooks.
        """
        template = self.templates.get("resolver_prompt.md", "")

        # Format evidence snippets
        evidence_lines = []
        if hasattr(context, "snippets") and context.snippets:
            for s in context.snippets:
                role_label = "Brand Agent" if getattr(s, "role", "") == "brand" else "Customer"
                type_label = getattr(s, "snippet_type", "turn").capitalize()
                tid = getattr(s, "thread_id", "")
                turn = getattr(s, "turn_id", "")
                evidence_lines.append(f"- [{role_label} ({type_label}) | Thread #{tid}-Turn{turn}]: {s.content}")
        else:
            evidence_lines.append("No prior conversation evidence retrieved.")
        evidence_str = "\n".join(evidence_lines)

        # Format policy steps
        policy_lines = []
        if hasattr(policy, "actionable_steps") and policy.actionable_steps:
            for idx, step in enumerate(policy.actionable_steps, start=1):
                policy_lines.append(f"{idx}. {step}")
        elif hasattr(policy, "diagnostic_steps") and policy.diagnostic_steps:
            for idx, step in enumerate(policy.diagnostic_steps, start=1):
                policy_lines.append(f"{idx}. {step}")
        else:
            policy_lines.append("No specific policy runbook steps defined.")
        policy_str = "\n".join(policy_lines)

        # Format canonical URLs
        url_lines = []
        if hasattr(policy, "canonical_urls") and policy.canonical_urls:
            for u in policy.canonical_urls:
                url_lines.append(f"- {u}")
        else:
            url_lines.append("None specified.")
        url_str = "\n".join(url_lines)

        # Variable substitution
        result = template.replace("{{CUSTOMER_MESSAGE}}", customer_message or "")
        result = result.replace("{{INTENT}}", intent or "UNKNOWN_OTHER")
        result = result.replace("{{EVIDENCE_SNIPPETS}}", evidence_str)
        result = result.replace("{{POLICY_STEPS}}", policy_str)
        result = result.replace("{{CANONICAL_URLS}}", url_str)

        return result
