import json
import unittest
from pathlib import Path

from app.agent import Agent
from app.llm import LLMError, extract_json
from app.scanner import scan, verify_evidence
from app.search import BM25
from app.segment import segment

LEASE = (Path(__file__).parent.parent / "samples" / "rental_agreement_bengaluru.txt").read_text(encoding="utf-8")


class FakeLLM:
    """Scripted stand-in for the model. `script` maps a predicate on the last
    message to a reply; anything else returns `default`."""
    model = "fake"
    configured = True

    def __init__(self, replies=None, scan_reply=None, summary_reply=None, fail=False):
        self.replies = list(replies or [])
        self.scan_reply, self.summary_reply, self.fail = scan_reply, summary_reply, fail
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        if self.fail:
            raise LLMError("boom")
        return self.replies.pop(0) if self.replies else '{"final": "done"}'

    def chat_json(self, messages, **kw):
        if self.fail:
            raise LLMError("boom")
        if "summarise" in messages[0]["content"]:
            return self.summary_reply or {"doc_type": "Rental agreement", "key_terms": [{"name": "Monthly rent", "value": "₹32,000", "clause_id": "C3"}, {"name": "x", "value": "y", "clause_id": "C999"}]}
        return self.scan_reply(messages) if self.scan_reply else {"findings": []}


class TestParsing(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure!\n```json\n{"a": [1,2]}\n```'), {"a": [1, 2]})
        self.assertEqual(extract_json('<think>maybe {broken</think> here: {"ok": true} trailing'), {"ok": True})
        self.assertIsNone(extract_json("no json at all"))


class TestSegmentAndSearch(unittest.TestCase):
    def test_lease_segments_by_heading(self):
        cs = segment(LEASE)
        self.assertEqual(len(cs), 17)
        self.assertTrue(cs[5]["title"].startswith("5. LOCK-IN PERIOD"))

    def test_paragraph_fallback(self):
        cs = segment("First paragraph is long enough to be a clause on its own.\n\nSecond paragraph is also long enough to count here.")
        self.assertEqual([c["id"] for c in cs], ["C1", "C2"])

    def test_section_and_article_headings(self):
        t = ("Section 1 Definitions\nWords mean things here and that is all there is to say.\n"
             "Section 2 Payment\nYou pay us the amount listed every month on the first day.\n"
             "Article III Termination\nEither party may end this with thirty days written notice.")
        self.assertEqual([c["title"] for c in segment(t)],
                         ["Section 1 Definitions", "Section 2 Payment", "Article III Termination"])

    def test_bm25(self):
        cs = segment(LEASE)
        top = BM25(cs).search("security deposit refunded", k=1)[0][0]
        self.assertIn("SECURITY DEPOSIT", top["text"])


class TestScanner(unittest.TestCase):
    def test_verify_evidence(self):
        clause = "Landlord and its agents may enter the premises at any time, with or without notice."
        self.assertTrue(verify_evidence("enter the premises at any time, with or without notice", clause))
        self.assertFalse(verify_evidence("tenant may sublet freely", clause))

    def test_scan_validates_model_output(self):
        def scan_reply(messages):
            return {"findings": [
                {"clause_id": "C9", "category": "entry_privacy", "risk": "HIGH", "title": "Entry without notice",
                 "evidence": "may enter and inspect the Premises at any time, with or without notice", "plain_english": "x", "suggested_revision": "y"},
                {"clause_id": "C9", "category": "entry_privacy", "risk": "high", "title": "dup"},
                {"clause_id": "C404", "category": "other", "risk": "high", "title": "hallucinated"},
                {"clause_id": "[C6]", "category": "lock_in_forfeiture", "risk": "extreme", "evidence": "invented words that are not there at all ok"},
            ]} if "[C9]" in messages[1]["content"] else {"findings": []}
        r = scan(segment(LEASE), FakeLLM(scan_reply=scan_reply))
        ids = [(f["clause_id"], f["risk"], f["verified"]) for f in r["findings"]]
        self.assertEqual(ids, [("C9", "high", True), ("C6", "medium", False)])
        self.assertEqual(r["summary"]["key_terms"][1]["clause_id"], None)
        self.assertFalse(r["degraded"])

    def test_key_term_ref_repaired(self):
        # Model cites section number 5 as C5 (deposit) instead of C6 (lock-in).
        summary = {"key_terms": [{"name": "Lock-in period", "value": "11 months; entire deposit forfeited if you vacate early", "clause_id": "C5"}]}
        r = scan(segment(LEASE), FakeLLM(summary_reply=summary))
        self.assertEqual(r["summary"]["key_terms"][0]["clause_id"], "C6")

    def test_scan_falls_back_to_rulebook(self):
        r = scan(segment(LEASE), FakeLLM(fail=True))
        self.assertTrue(r["degraded"])
        cats = {f["category"] for f in r["findings"]}
        self.assertTrue({"renewal_rent_hike", "arbitration_unilateral", "lock_in_forfeiture", "essential_services", "deposit"} <= cats)
        self.assertTrue(all(f["source"] == "rulebook" for f in r["findings"]))


class TestAgent(unittest.TestCase):
    def doc(self):
        cs = segment(LEASE)
        return {"clauses": cs, "index": BM25(cs), "findings": [], "summary": {"doc_type": "Rental agreement"}}

    def test_tool_loop_and_citations(self):
        llm = FakeLLM(replies=[
            json.dumps({"thought": "find it", "action": "search_clauses", "input": {"query": "vacate before lock-in forfeited"}}),
            "Here you go: " + json.dumps({"thought": "read", "action": "get_clause", "input": {"id": "c6"}}),
            json.dumps({"thought": "done", "final": "Your whole ₹3,20,000 deposit is forfeited [C6]. See also [C2] and [C99]."}),
        ])
        r = Agent(llm, self.doc()).run("What if I leave early?")
        self.assertEqual([t["action"] for t in r["trace"]], ["search_clauses", "get_clause", "final"])
        self.assertIn("LOCK-IN PERIOD", r["trace"][1]["observation"])
        self.assertEqual(r["citations"], ["C2", "C6", "C99"])
        self.assertIn("C99", r["unverified_citations"])
        self.assertNotIn("C6", r["unverified_citations"])

    def test_malformed_json_never_shown_raw(self):
        llm = FakeLLM(replies=[
            json.dumps({"action": "list_findings", "input": {}}),
            '{"thought": "the whole answer stuffed in here and then truncated',     # no final -> retry
            '{"thought": "ok", "final": "Negotiate the non-compete first [C8]. Ask for 6 months',  # truncated
        ])
        r = Agent(llm, self.doc()).run("what first?")
        self.assertEqual(r["answer"], "Negotiate the non-compete first [C8]. Ask for 6 months")
        self.assertFalse(r["answer"].startswith("{"))

    def test_normalize_citations(self):
        from app.agent import normalize_citations
        self.assertEqual(normalize_citations("fee【C10】 and 【C11†L3】, [C2, C5], (C7), Clause C8, [C3], bare C4. ABC12"),
                         "fee[C10] and [C11], [C2] [C5], [C7], [C8], [C3], bare [C4]. ABC12")

    def test_india_context_and_language_in_prompt(self):
        llm = FakeLLM(replies=[json.dumps({"final": "ठीक है [C3]"})])
        Agent(llm, self.doc()).run("rent?", language="Hindi")
        system = llm.calls[0][0]["content"]
        self.assertIn("Jurisdiction: INDIA", system)
        self.assertIn("Write the final answer in Hindi", system)
        Agent(llm, self.doc()).run("rent?", language="Klingon")
        self.assertIn("Write the final answer in English", llm.calls[-1][0]["content"])

    def test_bad_tool_and_step_budget(self):
        llm = FakeLLM(replies=[json.dumps({"action": "delete_everything", "input": {}})] * 6 + ["Final prose answer."])
        r = Agent(llm, self.doc()).run("hi")
        self.assertIn("Unknown tool", r["trace"][0]["observation"])
        self.assertEqual(r["answer"], "Final prose answer.")


class TestAPI(unittest.TestCase):
    def test_end_to_end(self):
        from fastapi.testclient import TestClient
        from app import main
        main.llm = FakeLLM(replies=[json.dumps({"final": "Rent is ₹32,000 [C3]."})])
        c = TestClient(main.app)
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(len(c.get("/api/samples").json()), 3)
        r = c.post("/api/analyze", json={"text": LEASE}).json()
        self.assertEqual(len(r["clauses"]), 17)
        a = c.post("/api/ask", json={"doc_id": r["doc_id"], "question": "rent?"}).json()
        self.assertEqual(a["citations"], ["C3"])
        self.assertEqual(c.post("/api/ask", json={"doc_id": "nope", "question": "rent?"}).status_code, 404)
        self.assertEqual(c.post("/api/extract", content=b"plain text body").json()["text"], "plain text body")


if __name__ == "__main__":
    unittest.main()
