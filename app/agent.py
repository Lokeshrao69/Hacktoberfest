"""A small, model-agnostic agent harness.

Native tool-calling support varies a lot across open-weight models and serving
stacks, so the harness uses a plain JSON action protocol that any instruction-
tuned model can follow, with parse repair, a step budget, and citation checks.
"""
import json
import re

from . import rules
from .llm import LLMError, extract_json

MAX_STEPS = 6
LANGUAGES = ("English", "Hindi", "Tamil", "Telugu", "Kannada", "Marathi", "Bengali", "Malayalam", "Gujarati")

SYSTEM = """You are ClauseGuard, an assistant that answers questions about ONE contract for the person who has to sign it.
You cannot see the contract directly; use tools to read it. Every turn, reply with exactly ONE JSON object and nothing else:

"thought" is ONE short sentence; the full answer goes only in "final".
To use a tool:  {{"thought": "why", "action": "<tool>", "input": {{...}}}}
To answer:      {{"thought": "why", "final": "your answer in markdown"}}

Tools:
- search_clauses  input {{"query": "keywords", "k": 4}}   -> most relevant clauses (BM25)
- get_clause      input {{"id": "C7"}}                     -> full text of one clause
- list_findings   input {{}}                               -> risks found by the scan
- lookup_rule     input {{"topic": "auto renewal"}}        -> what is fair/standard for a topic

Rules:
- Read the relevant clauses before answering. Cite clauses inline in square brackets like [C7]; only cite clauses you have read.
- Always refer to clauses by their id (C7), never as "Clause 7" or by the section number printed in the text; they differ.
- Quote exact numbers, dates and amounts. If the contract does not say, say so.
- When useful, propose concrete wording the signer could ask for.
- Keep the answer under 200 words. You are not a lawyer; for high-stakes situations suggest a lawyer or free legal aid (your District Legal Services Authority / NALSA, helpline 15100).

{region}
Write the final answer in {language}; keep clause ids like [C7], numbers and ₹ amounts unchanged.
Document: {doc_type}, {n} clauses ({first}..{last})."""


def normalize_citations(text: str) -> str:
    """Rewrite model citation styles (【C3】, (C3), clause C3, [C3, C5]) as [C3]."""
    text = re.sub(r"【\s*(C\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\[(C\d+(?:\s*[,;]\s*C\d+)+)\]",
                  lambda m: " ".join(f"[{c.strip()}]" for c in re.split(r"[,;]", m.group(1))), text)
    text = re.sub(r"\((C\d+)\)", r"[\1]", text)
    text = re.sub(r"\b(?:clause|section)\s+(C\d+)\b", r"[\1]", text, flags=re.I)
    text = re.sub(r"(?<![\[\w])(C\d+)(?![\]\w])", r"[\1]", text)  # bare C3
    return text


_FINAL = re.compile(r'"final"\s*:\s*"((?:[^"\\]|\\.)*)', re.S)


def salvage_final(reply: str):
    """Recover the "final" string from malformed or truncated JSON."""
    m = _FINAL.search(reply or "")
    if not m or len(m.group(1)) < 20:
        return None
    raw = m.group(1).rstrip("\\")
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw.replace("\\n", "\n").replace('\\"', '"')


class Agent:
    def __init__(self, llm, doc):
        self.llm = llm
        self.doc = doc  # {"clauses", "index", "findings", "summary"}
        self.by_id = {c["id"]: c for c in doc["clauses"]}
        self.seen = set()

    # ---- tools ----
    def t_search_clauses(self, query="", k=4):
        hits = self.doc["index"].search(str(query), k=max(1, min(int(k or 4), 6)))
        if not hits:
            return "No matching clauses. Try different keywords or list_findings."
        out = []
        for c, s in hits:
            self.seen.add(c["id"])
            out.append(f"[{c['id']}] {c['text'][:700]}")
        return "\n\n".join(out)

    def t_get_clause(self, id=""):
        cid = str(id).strip().strip("[]").upper()
        c = self.by_id.get(cid)
        if not c:
            return f"No clause {cid}. Valid ids: C1..C{len(self.by_id)}."
        self.seen.add(cid)
        return f"[{cid}] {c['text']}"

    def t_list_findings(self):
        fs = self.doc.get("findings", [])
        if not fs:
            return "The scan found no risky clauses."
        for f in fs:
            self.seen.add(f["clause_id"])
        return "\n".join(f"[{f['clause_id']}] {f['risk'].upper()} {f['title']}: {f['plain_english']}" for f in fs)

    def t_lookup_rule(self, topic=""):
        found = rules.lookup(str(topic))
        if not found:
            return "No rulebook entry. Known topics: " + ", ".join(rules.RULES_BY_ID)
        return "\n".join(f"{r['name']}: risk = {r['why']} Fair standard = {r['fair_standard']}" for r in found)

    TOOLS = {
        "search_clauses": t_search_clauses,
        "get_clause": t_get_clause,
        "list_findings": t_list_findings,
        "lookup_rule": t_lookup_rule,
    }

    def _call(self, action, args):
        fn = self.TOOLS.get(action)
        if not fn:
            return f"Unknown tool '{action}'. Available: {', '.join(self.TOOLS)}"
        if not isinstance(args, dict):
            args = {}
        try:
            return fn(self, **args)
        except TypeError as e:
            return f"Bad input for {action}: {e}"

    # ---- loop ----
    def run(self, question, history=None, language="English"):
        clauses = self.doc["clauses"]
        system = SYSTEM.format(
            doc_type=self.doc.get("summary", {}).get("doc_type", "contract"),
            n=len(clauses), first=clauses[0]["id"] if clauses else "C1",
            last=clauses[-1]["id"] if clauses else "C1",
            region=rules.region_note(),
            language=language if language in LANGUAGES else "English",
        )
        messages = [{"role": "system", "content": system}]
        for turn in (history or [])[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:2000]})
        messages.append({"role": "user", "content": question})

        trace, answer = [], None
        for step in range(MAX_STEPS):
            reply = self.llm.chat(messages, temperature=0.1, max_tokens=4096)
            obj = extract_json(reply)
            if not isinstance(obj, dict):
                salvaged = salvage_final(reply)
                if salvaged:
                    answer = salvaged  # broken/truncated JSON, but the answer field survived
                    trace.append({"thought": "(repaired malformed JSON reply)", "action": "final"})
                    break
                looks_like_json = reply.lstrip().startswith(("{", "```"))
                if reply.strip() and step > 0 and not looks_like_json:
                    answer = reply.strip()  # model answered in prose; accept it
                    break
                messages += [{"role": "assistant", "content": reply[:1500]},
                             {"role": "user", "content": 'That was not one valid JSON object. Reply again with ONE short JSON object: a one-sentence "thought" plus either "action"/"input" or "final".'}]
                trace.append({"thought": "(unparseable reply, asked to retry)"})
                continue
            if "final" in obj:
                answer = str(obj["final"])
                trace.append({"thought": obj.get("thought", ""), "action": "final"})
                break
            action, args = obj.get("action", ""), obj.get("input", {})
            observation = self._call(action, args)
            trace.append({"thought": obj.get("thought", ""), "action": action, "input": args,
                          "observation": observation[:600]})
            messages += [{"role": "assistant", "content": json.dumps(obj)},
                         {"role": "user", "content": f"Observation:\n{observation}"}]

        if answer is None:
            messages.append({"role": "user", "content": "Step budget reached. Give your final answer now as plain markdown."})
            try:
                answer = self.llm.chat(messages, temperature=0.1).strip() or "I couldn't finish."
            except LLMError:
                answer = "I ran out of steps before finishing. Try a more specific question."

        answer = normalize_citations(answer)
        cited = sorted(set(re.findall(r"\[(C\d+)\]", answer)), key=lambda x: int(x[1:]))
        unverified = [c for c in cited if c not in self.seen or c not in self.by_id]
        return {"answer": answer, "trace": trace, "citations": cited, "unverified_citations": unverified}
