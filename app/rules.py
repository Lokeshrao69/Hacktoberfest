"""Rulebook: curated red-flag categories that ground the model's judgement."""
import json
import os
import re
from pathlib import Path

RULES = json.loads((Path(__file__).parent / "data" / "rules.json").read_text(encoding="utf-8"))
RULES_BY_ID = {r["id"]: r for r in RULES}
_COMPILED = [(r["id"], [re.compile(p, re.I) for p in r["patterns"]]) for r in RULES]


def flag(text: str):
    """Rule ids whose patterns match the text (cheap prior for the model)."""
    return [rid for rid, pats in _COMPILED if any(p.search(text) for p in pats)]


def lookup(topic: str):
    """Find rules by id or fuzzy topic words."""
    topic = topic.lower().strip()
    if topic in RULES_BY_ID:
        return [RULES_BY_ID[topic]]
    words = set(re.findall(r"[a-z]+", topic))
    scored = []
    for r in RULES:
        hay = set(re.findall(r"[a-z]+", (r["id"] + " " + r["name"] + " " + r["why"]).lower()))
        overlap = len(words & hay)
        if overlap:
            scored.append((overlap, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:2]]


INDIA_NOTE = """Jurisdiction: INDIA. The signer is in India; money is in Indian Rupees (write amounts as ₹ with Indian grouping, e.g. ₹3,20,000; 1 lakh = ₹1,00,000, 1 crore = ₹1,00,00,000; "Rs." means ₹).
Judge clauses against Indian law and market practice, for example: Model Tenancy Act 2021 (deposit cap 2 months residential, 24h entry notice, no cutting essential supplies, landlord does structural repairs), Indian Contract Act 1872 (s.27 post-exit non-competes generally void; s.74 only reasonable compensation, not penalties), Arbitration and Conciliation Act 1996 (no unilateral appointment of a sole arbitrator by an interested party), Registration Act 1908 (leases of 12+ months must be registered), Code on Wages 2019, Shops and Establishments Acts, DPDP Act 2023, Consumer Protection Act 2019, MSMED Act 2006 (45-day payment to registered MSEs).
Name the relevant Indian law briefly in plain_english when it matters, and say when a clause is likely unenforceable. Note when a model law (like the Model Tenancy Act) applies only in states that adopted it."""


def region_note():
    region = os.getenv("CONTRACT_REGION", "India").strip()
    if region.lower() in ("india", "in"):
        return INDIA_NOTE
    return f"Jurisdiction: {region}. Judge clauses against the law and market practice of {region}."


def catalog_text():
    return "\n".join(f"- {r['id']}: {r['name']} (fair standard: {r['fair_standard']})" for r in RULES)
