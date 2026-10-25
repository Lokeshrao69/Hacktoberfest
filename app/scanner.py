"""Risk scan: rulebook prior -> open-weight LLM judgement -> evidence verification."""
import math
import re
from concurrent.futures import ThreadPoolExecutor

from . import rules
from .llm import LLMError
from .search import tokenize

BATCH_CHARS = 2200  # small batches run in parallel and stay well under request timeouts
RISK_POINTS = {"high": 18, "medium": 8, "low": 2}

SCAN_SYSTEM = """You are ClauseGuard, a careful contract reviewer protecting the person who has to SIGN this document (tenant, customer, employee, freelancer).
Review each clause and report ONLY clauses that are unfavourable, unusual or risky for the signer. Skip boilerplate that is fair.

Rulebook categories (use the id as "category"; use "other" if none fit):
{catalog}

Return ONLY JSON in this shape:
{{"findings": [
  {{"clause_id": "C3",
    "category": "auto_renewal",
    "risk": "high" | "medium" | "low",
    "title": "short headline, max 8 words",
    "evidence": "EXACT words copied from the clause (max 30 words)",
    "plain_english": "what this means for the signer, 1-2 sentences, no legalese",
    "suggested_revision": "fairer replacement wording the signer could propose"}}
]}}
Rules: clause_id must be one of the ids given. evidence must be copied verbatim. Be specific with numbers, dates and amounts."""

SUMMARY_SYSTEM = """You summarise contracts for the person who has to sign them. Return ONLY JSON:
{"doc_type": "e.g. Residential lease", "parties": ["..."], "summary": "2-3 plain-English sentences",
 "key_terms": [{"name": "e.g. Monthly rent", "value": "e.g. $2,100", "clause_id": "C2"}],
 "questions_to_ask": ["3 short questions the signer should ask before signing"]}
Include up to 8 key terms (money, dates, durations, notice periods). Only use clause ids that appear in the text."""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def verify_evidence(evidence: str, clause_text: str) -> bool:
    """True if the quoted evidence really appears in the clause (whitespace/punctuation-insensitive),
    or at least 80% of its words appear in order-insensitive form for long quotes."""
    ev, cl = _norm(evidence or ""), _norm(clause_text)
    if not ev:
        return False
    if ev in cl:
        return True
    words = ev.split()
    if len(words) < 6:
        return False
    cl_words = set(cl.split())
    return sum(w in cl_words for w in words) / len(words) >= 0.8


def _fix_ref(term, clauses, by_id):
    """Models often confuse section numbers ("10.") with clause ids (C11). Keep the cited
    clause only if it supports the term's value best; otherwise re-point to the clause that does."""
    words = set(tokenize(f"{term.get('value', '')} {term.get('name', '')}"))
    if not words:
        return term.get("clause_id") if term.get("clause_id") in by_id else None

    def support(c):
        return len(words & set(tokenize(c["text"]))) / len(words)

    best = max(clauses, key=support)
    cited = by_id.get(term.get("clause_id"))
    if cited and support(cited) >= support(best):
        return cited["id"]
    return best["id"] if support(best) > 0 else None


def _batches(clauses):
    batch, size = [], 0
    for c in clauses:
        if batch and size + len(c["text"]) > BATCH_CHARS:
            yield batch
            batch, size = [], 0
        batch.append(c)
        size += len(c["text"])
    if batch:
        yield batch


def _scan_batch(llm, batch):
    parts = []
    for c in batch:
        hint = ", ".join(c["flags"]) or "none"
        parts.append(f"[{c['id']}] (rulebook hints: {hint})\n{c['text']}")
    messages = [
        {"role": "system", "content": SCAN_SYSTEM.format(catalog=rules.catalog_text())},
        {"role": "user", "content": "Clauses to review:\n\n" + "\n\n---\n\n".join(parts)},
    ]
    data = llm.chat_json(messages)
    items = data.get("findings", []) if isinstance(data, dict) else data
    return items if isinstance(items, list) else []


def _rulebook_findings(batch):
    """Fallback when the model is unavailable: pure rulebook matches."""
    out = []
    for c in batch:
        for rid in c["flags"]:
            r = rules.RULES_BY_ID[rid]
            out.append({
                "clause_id": c["id"], "category": rid, "risk": r["severity"], "title": r["name"],
                "evidence": "", "plain_english": r["why"], "suggested_revision": r["fair_standard"],
                "source": "rulebook",
            })
    return out


def _clean(raw, by_id):
    """Validate model output: real clause ids, known risk levels, verified quotes."""
    out, seen = [], set()
    for f in raw:
        if not isinstance(f, dict):
            continue
        cid = str(f.get("clause_id", "")).strip().strip("[]")
        if cid not in by_id:
            continue  # hallucinated clause id
        risk = str(f.get("risk", "medium")).lower()
        risk = risk if risk in RISK_POINTS else "medium"
        cat = f.get("category") if f.get("category") in rules.RULES_BY_ID else "other"
        key = (cid, cat)
        if key in seen:
            continue
        seen.add(key)
        evidence = str(f.get("evidence", ""))
        out.append({
            "clause_id": cid,
            "category": cat,
            "risk": risk,
            "title": str(f.get("title", ""))[:80] or rules.RULES_BY_ID.get(cat, {}).get("name", "Issue"),
            "evidence": evidence,
            "verified": verify_evidence(evidence, by_id[cid]["text"]),
            "plain_english": str(f.get("plain_english", "")),
            "suggested_revision": str(f.get("suggested_revision", "")),
            "source": f.get("source", "model"),
        })
    return out


def score(findings) -> int:
    pts = sum(RISK_POINTS[f["risk"]] * (1 if f.get("verified", True) else 0.5) for f in findings)
    # Diminishing returns so a handful of issues reads as moderate and only
    # contracts riddled with high-risk terms approach 100.
    return round(100 * (1 - math.exp(-pts / 90)))


def scan(clauses, llm):
    for c in clauses:
        c["flags"] = rules.flag(c["text"])
    by_id = {c["id"]: c for c in clauses}
    batches = list(_batches(clauses))
    errors = []

    def run(batch):
        try:
            return _clean(_scan_batch(llm, batch), by_id)
        except (LLMError, AttributeError, TypeError) as e:
            errors.append(str(e))
            return _clean(_rulebook_findings(batch), by_id)

    def summarise():
        text = "\n\n".join(f"[{c['id']}] {c['text']}" for c in clauses)[:14000]
        try:
            s = llm.chat_json([
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": text},
            ])
            return s if isinstance(s, dict) else {}
        except LLMError as e:
            errors.append(str(e))
            return {}

    with ThreadPoolExecutor(max_workers=6) as pool:
        summary_future = pool.submit(summarise)
        findings = [f for fs in pool.map(run, batches) for f in fs]
        summary = summary_future.result()

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f["risk"]], int(f["clause_id"][1:])))
    summary["key_terms"] = [k for k in summary.get("key_terms", []) if isinstance(k, dict)]
    for k in summary["key_terms"]:
        k["clause_id"] = _fix_ref(k, clauses, by_id)
    return {
        "summary": summary,
        "findings": findings,
        "risk_score": score(findings),
        "degraded": bool(errors),
        "errors": errors[:3],
    }
