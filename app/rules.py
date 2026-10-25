"""Rulebook: curated red-flag categories that ground the model's judgement."""
import json
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


def catalog_text():
    return "\n".join(f"- {r['id']}: {r['name']} (fair standard: {r['fair_standard']})" for r in RULES)
