"""Command-line ClauseGuard, also the entry point for the agent skill.

    python -m app.cli contract.txt                 # risk report (markdown)
    python -m app.cli contract.pdf --ask "Can I sublet?"
    python -m app.cli contract.txt --json          # machine-readable
"""
import argparse
import json
import sys
from pathlib import Path

from .agent import LANGUAGES, Agent
from .llm import LLM, LLMError, load_dotenv
from .scanner import scan
from .search import BM25
from .segment import segment


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def report_md(result, clauses) -> str:
    s = result["summary"]
    out = [f"# ClauseGuard report: {s.get('doc_type', 'Contract')}",
           f"**Risk score:** {result['risk_score']}/100 · {len(clauses)} clauses · {len(result['findings'])} findings", ""]
    if s.get("summary"):
        out += [s["summary"], ""]
    if s.get("key_terms"):
        out += ["## Key terms"] + [f"- **{k.get('name')}**: {k.get('value')}" + (f" [{k['clause_id']}]" if k.get("clause_id") else "") for k in s["key_terms"]] + [""]
    out.append("## Risks")
    for f in result["findings"]:
        out.append(f"### [{f['risk'].upper()}] {f['title']} [{f['clause_id']}]")
        out.append(f["plain_english"])
        if f["evidence"]:
            out.append(f"> \"{f['evidence']}\" ({'verified' if f['verified'] else 'NOT verified verbatim'})")
        if f["suggested_revision"]:
            out.append(f"**Ask for instead:** {f['suggested_revision']}")
        out.append("")
    if s.get("questions_to_ask"):
        out += ["## Ask before signing"] + [f"- {q}" for q in s["questions_to_ask"]]
    if result["degraded"]:
        out.append("\n_Note: part of the model review failed; some findings are rulebook-only._")
    return "\n".join(out)


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p =argparse.ArgumentParser(description="Flag risky clauses in a contract with an open-weight LLM.")
    p.add_argument("file", type=Path)
    p.add_argument("--ask", help="ask a question about the contract instead of printing the report")
    p.add_argument("--json", action="store_true", help="print JSON")
    p.add_argument("--lang", default="English", choices=LANGUAGES, help="language for --ask answers")
    args = p.parse_args(argv)

    llm = LLM()
    if not llm.configured:
        sys.exit("Set MODEL_ACCESS_KEY (DigitalOcean) or LLM_BASE_URL (self-hosted model).")
    clauses = segment(read_text(args.file))
    if not clauses:
        sys.exit("No clauses found in the document.")
    try:
        result = scan(clauses, llm)
        if args.ask:
            doc = {"clauses": clauses, "index": BM25(clauses), **result}
            ans = Agent(llm, doc).run(args.ask, language=args.lang)
            print(json.dumps(ans, indent=2) if args.json else ans["answer"])
        else:
            print(json.dumps({"clauses": clauses, **result}, indent=2) if args.json else report_md(result, clauses))
    except LLMError as e:
        sys.exit(f"Model error: {e}")


if __name__ == "__main__":
    main()
