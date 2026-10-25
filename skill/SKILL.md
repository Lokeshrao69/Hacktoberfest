---
name: clauseguard
description: Review a contract, lease, offer letter, freelance agreement or terms of service for clauses that are risky for the signer, using an open-weight model on DigitalOcean. Use when the user asks "is this contract fair", "what should I watch out for before signing", or asks a question about a specific agreement file.
---

# ClauseGuard skill

ClauseGuard segments a contract into clauses (C1, C2, …), checks them against a rulebook of 14 red-flag categories, has an open-weight LLM review each clause, verifies every quoted piece of evidence against the source, and answers questions with clause citations.

## Setup (once)

From the ClauseGuard repo root:

```bash
pip install -r requirements.txt
export MODEL_ACCESS_KEY=...          # DigitalOcean Gradient model access key
# optional: export LLM_MODEL=openai-gpt-oss-120b  (or LLM_BASE_URL for a self-hosted vLLM/Ollama)
```

## Use

- Full risk report as markdown: `python -m app.cli path/to/contract.pdf`
- A specific question: `python -m app.cli path/to/contract.txt --ask "What happens if I move out early?"`
- Structured output for further processing: add `--json`

## How to present results

1. Lead with the risk score and the HIGH findings.
2. Always keep clause ids like [C7] so the user can check the source.
3. If a finding says "NOT verified verbatim", open that clause and confirm it before repeating the claim.
4. Offer the "Ask for instead" wording as negotiation language.
5. Say that this is information, not legal advice. For high stakes (employment, large sums, eviction), suggest a lawyer.
