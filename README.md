# ClauseGuard 🛡️

**Read the fine print before you sign.** ClauseGuard is an AI agent that reviews leases, freelance contracts, job offers and terms of service *for the person signing them*. It flags risky clauses, explains them in plain English, drafts fairer wording to negotiate with, and answers questions about the contract with citations to the exact clause.

It runs on **open-weight models** (default: `openai-gpt-oss-120b`) served by **DigitalOcean Serverless Inference**. The same code runs on a self-hosted model on a DigitalOcean GPU Droplet.

## Why

Most people sign contracts they can't fully parse: 90-day auto-renewal windows, "all remaining rent becomes due", one-way indemnities, non-competes. Lawyers are expensive. Generic chatbots hallucinate clauses that aren't there. ClauseGuard is built so you can check every claim it makes.

## How the open-weight model is used

```
 contract ─► 1. Segment ─► 2. Rulebook prior ─► 3. Open-weight LLM review ─► 4. Verify ─► report
             C1..Cn        14 red-flag            per-clause JSON findings      quotes checked
             clauses       categories (regex)     + plain English + revision    against source
                                                                                    │
 question ─► Agent harness (JSON action protocol) ── tools: search_clauses (BM25),  │
             max 6 steps, parse repair            get_clause, list_findings,  ◄─────┘
                                                  lookup_rule
             ─► answer with [C#] citations ─► citation check (only clauses the agent actually read)
```

* **A model-agnostic agent harness** (`app/agent.py`). Native tool calling varies a lot between open-weight models and serving stacks, so the harness uses a plain JSON action protocol that any instruction-tuned model can follow. It handles models that wrap JSON in prose, code fences or `<think>` blocks, and it repairs bad output, enforces a step budget and rejects unknown tools.
* **Grounding and verification** (`app/scanner.py`):
  * Every finding must quote the clause verbatim. The quote is checked against the source text and marked ✓ verified or ⚠ not verified.
  * Hallucinated clause ids are dropped.
  * Any `[C#]` the agent cites without having read that clause is flagged.
  * The risk score is computed deterministically from verified findings, not taken from the model.
* **Graceful degradation**: if the model fails, findings fall back to the curated rulebook (`app/data/rules.json`) so the user still gets a report.
* **Batched, parallel review** keeps latency low on long documents.

## Three ways to use it

| Surface | Command |
|---|---|
| Web app | `uvicorn app.main:app` → http://localhost:8000 |
| CLI | `python -m app.cli samples/apartment_lease.txt` / `--ask "Can I sublet?"` / `--json` |
| Agent skill | `skill/SKILL.md` lets coding agents (Claude Code, etc.) call ClauseGuard as a skill |

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # paste your DigitalOcean model access key into MODEL_ACCESS_KEY
uvicorn app.main:app --reload
```

Get a key in the DigitalOcean control panel: **Gradient AI Platform → Serverless Inference → Model Access Keys**. To use a different open-weight model, set `LLM_MODEL`, e.g. `openai-gpt-oss-20b`, `llama-4-maverick`, `deepseek-3.2` or `qwen3.5-397b-a17b`.

Tests run without a key (a scripted fake model stands in):

```bash
python -m unittest discover -s tests -t . -v
```

## Deploy on DigitalOcean

### App Platform + Serverless Inference (recommended)

1. Push this folder to a GitHub repo.
2. Edit `.do/app.yaml` → `github.repo`.
3. `doctl apps create --spec .do/app.yaml`, or in the control panel **Create App → GitHub repo**, which detects the Dockerfile.
4. Set `MODEL_ACCESS_KEY` as an encrypted environment variable.

### Fully self-hosted on a GPU Droplet

```bash
# on the GPU Droplet
pip install vllm && vllm serve openai/gpt-oss-20b --port 8000
# on the app
LLM_BASE_URL=http://<droplet-ip>:8000/v1  LLM_MODEL=openai/gpt-oss-20b
```

Ollama works too: `LLM_BASE_URL=http://<host>:11434/v1`.

## Project layout

```
app/
  llm.py        OpenAI-compatible client + tolerant JSON extraction
  segment.py    clause segmentation (numbered / ALL-CAPS headings, paragraph fallback)
  rules.py      rulebook loader + pattern matching   (data/rules.json: 14 categories)
  scanner.py    batched LLM review, validation, evidence verification, scoring
  search.py     dependency-free BM25 over clauses
  agent.py      JSON-protocol agent harness with 4 tools + citation checks
  main.py       FastAPI API + static UI
  cli.py        command line / skill entry point
static/index.html   single-file UI (light/dark, mobile)
samples/            two original sample contracts with planted red flags
skill/SKILL.md      agent skill definition
tests/              unit + end-to-end tests with a fake model
```

## Demo script (2 minutes)

1. Click **apartment lease**, then **Analyse contract**. The risk score appears, and the HIGH findings show the 90-day auto-renewal, "all remaining rent due" and landlord-picked arbitration.
2. Click a finding to jump to the highlighted clause. Point out the "✓ quote verified in source" badge.
3. Ask *"What happens if I move out early?"* Expand **Agent trace** to show the `search_clauses` → `get_clause` tool calls, then click the `[C11]` citation.
4. Open **Ask before signing** for the negotiation checklist.

*ClauseGuard provides information, not legal advice.*
