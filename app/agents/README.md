# Multi-Agent Conversational Support Architecture

This directory defines the interface contracts, lifecycle protocols, and prompt specifications for the Hiver AI Customer Support Agent.

---

## 1. Architectural Philosophy

Customer support is fundamentally a **multi-stage decision problem**:
1. Understanding the user's emotional state and intent (**Triage**).
2. Grounding the response in factual company policies and technical documentation (**Policy Retrieval**).
3. Synthesizing an empathetic, context-aware reply in the brand's persona (**Resolution**).
4. Verifying correctness, PII privacy, safety, and confidence before dispatching (**Escalation Guardrails**).

Monolithic single-prompt LLMs fail in production because they blur the distinction between factual verification and creative text generation, leading to hallucinations and compliance breaches. Our pipeline separates concerns across four specialized agents adhering to the abstract interfaces in `interfaces.py`.

---

## 2. Agent Responsibilities & Contracts

### A. Triage & Intent Agent (`ITriageAgent`)
* **Role**: Primary intake filter.
* **Responsibilities**:
  - Language detection (filtering non-English queries).
  - Sentiment & frustration quantification (-1.0 to +1.0).
  - Multi-class intent classification over Spotify domain taxonomy (e.g., `PLAYBACK_OFFLINE_ERROR`, `BILLING_SUBSCRIPTION`, `ACCOUNT_LOGIN_ACCESS`, `PLAYLIST_LIBRARY_SYNC`).
  - Entity extraction (OS version, device type, Spotify app build).
* **Output**: `TriageResult`

### B. Policy & Knowledge Agent (`IPolicyKnowledgeAgent`)
* **Role**: Grounded knowledge retriever.
* **Responsibilities**:
  - Query reformulation and dense/hybrid retrieval against indexed Spotify support documentation, community runbooks, and known service outages.
  - Extraction of canonical links (`<LINK:HELP_PORTAL>`).
  - Assessment of whether the query requires authenticated account access (PII) or can be resolved via public troubleshooting steps.
* **Output**: `PolicyContext`

### C. Resolver Agent (`IResolverAgent`)
* **Role**: Conversational synthesis and dialogue management.
* **Responsibilities**:
  - Multi-turn state tracking across the recent turn horizon ($K \le 4$).
  - Generation of structured, step-by-step diagnostic workflows.
  - Maintaining the friendly, tech-savvy, and proactive Spotify brand tone.
  - Ensuring the generated output is strictly grounded in the retrieved `PolicyContext`.
* **Output**: `AgentResponse`

### D. Safety & Escalation Agent (`IEscalationAgent`)
* **Role**: Operational safety gate and human-in-the-loop router.
* **Responsibilities**:
  - PII leakage detection (blocking credit card numbers, passwords, or personal identity details).
  - Toxic / abusive language policy enforcement.
  - Confidence scoring and hallucination detection (faithfulness metric).
  - Routing low-confidence or high-severity cases to appropriate human queues (`human_support`, `billing_tier2`, `trust_and_safety`).
* **Output**: `EscalationDecision`

---

## 3. Orchestration Workflow

```
[Inbound Tweet]
       │
       ▼
┌──────────────┐
│ Triage Agent │ ──(Intent, Sentiment, Entities)──┐
└──────────────┘                                  │
                                                  ▼
                                       ┌──────────────────────┐
                                       │ Policy/Knowledge     │
                                       │ Agent                │
                                       └──────────────────────┘
                                                  │
                                            (Policy Docs &
                                             Runbook Steps)
                                                  │
                                                  ▼
                                       ┌──────────────────────┐
                                       │ Resolver Agent       │
                                       └──────────────────────┘
                                                  │
                                          (Proposed Response)
                                                  │
                                                  ▼
                                       ┌──────────────────────┐
                                       │ Safety & Escalation  │
                                       │ Agent                │
                                       └──────────────────────┘
                                                  │
                       ┌──────────────────────────┴──────────────────────────┐
                       ▼                                                     ▼
              [Safe & High Conf]                                    [Escalate / Low Conf]
                       │                                                     │
                       ▼                                                     ▼
           {Publish Public Tweet}                                   {Handoff to Hiver Inbox}
```
