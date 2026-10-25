import json
import unittest
from pathlib import Path

from app.agent import Agent
from app.llm import LLMError, extract_json
from app.scanner import scan, verify_evidence
from app.search import BM25
from app.segment import segment

LEASE = (Path(__file__).parent.parent / "samples" / "apartment_lease.txt").read_text(encoding="utf-8")


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
            return self.summary_reply or {"doc_type": "Lease", "key_terms": [{"name": "Rent", "value": "$2,150", "clause_id": "C3"}, {"name": "x", "value": "y", "clause_id": "C999"}]}
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
        self.assertTrue(cs[10]["title"].startswith("10. EARLY TERMINATION"))

    def test_paragraph_fallback(self):
        cs = segment("First paragraph is long enough to be a clause on its own.\n\nSecond paragraph is also long enough to count here.")
        self.assertEqual([c["id"] for c in cs], ["C1", "C2"])

    def test_bm25(self):
        cs = segment(LEASE)
        top = BM25(cs).search("security deposit returned", k=1)[0][0]
        self.assertIn("SECURITY DEPOSIT", top["text"])


class TestScanner(unittest.TestCase):
    def test_verify_evidence(self):
        clause = "Landlord and its agents may enter the premises at any time, with or without notice."
        self.assertTrue(verify_evidence("enter the premises at any time, with or without notice", clause))
        self.assertFalse(verify_evidence("tenant may sublet freely", clause))

    def test_scan_validates_model_output(self):
        def scan_reply(messages):
            return {"findings": [
                {"clause_id": "C8", "category": "entry_privacy", "risk": "HIGH", "title": "Entry without notice",
                 "evidence": "may enter the premises at any time, with or without notice", "plain_english": "x", "suggested_revision": "y"},
                {"clause_id": "C8", "category": "entry_privacy", "risk": "high", "title": "dup"},
                {"clause_id": "C404", "category": "other", "risk": "high", "title": "hallucinated"},
                {"clause_id": "[C11]", "category": "early_termination", "risk": "extreme", "evidence": "invented words that are not there at all ok"},
            ]} if "[C8]" in messages[1]["content"] else {"findings": []}
        r = scan(segment(LEASE), FakeLLM(scan_reply=scan_reply))
        ids = [(f["clause_id"], f["risk"], f["verified"]) for f in r["findings"]]
        self.assertEqual(ids, [("C8", "high", True), ("C11", "medium", False)])
        self.assertEqual(r["summary"]["key_terms"][1]["clause_id"], None)
        self.assertFalse(r["degraded"])

    def test_scan_falls_back_to_rulebook(self):
        r = scan(segment(LEASE), FakeLLM(fail=True))
        self.assertTrue(r["degraded"])
        cats = {f["category"] for f in r["findings"]}
        self.assertTrue({"auto_renewal", "arbitration_class_waiver", "early_termination"} <= cats)
        self.assertTrue(all(f["source"] == "rulebook" for f in r["findings"]))


class TestAgent(unittest.TestCase):
    def doc(self):
        cs = segment(LEASE)
        return {"clauses": cs, "index": BM25(cs), "findings": [], "summary": {"doc_type": "Lease"}}

    def test_tool_loop_and_citations(self):
        llm = FakeLLM(replies=[
            json.dumps({"thought": "find it", "action": "search_clauses", "input": {"query": "vacate early termination"}}),
            "Here you go: " + json.dumps({"thought": "read", "action": "get_clause", "input": {"id": "c11"}}),
            json.dumps({"thought": "done", "final": "You owe all remaining rent plus two months [C11]. See also [C2] and [C99]."}),
        ])
        r = Agent(llm, self.doc()).run("What if I leave early?")
        self.assertEqual([t["action"] for t in r["trace"]], ["search_clauses", "get_clause", "final"])
        self.assertIn("EARLY TERMINATION", r["trace"][1]["observation"])
        self.assertEqual(r["citations"], ["C2", "C11", "C99"])
        self.assertIn("C99", r["unverified_citations"])
        self.assertNotIn("C11", r["unverified_citations"])

    def test_bad_tool_and_step_budget(self):
        llm = FakeLLM(replies=[json.dumps({"action": "delete_everything", "input": {}})] * 6 + ["Final prose answer."])
        r = Agent(llm, self.doc()).run("hi")
        self.assertIn("Unknown tool", r["trace"][0]["observation"])
        self.assertEqual(r["answer"], "Final prose answer.")


class TestAPI(unittest.TestCase):
    def test_end_to_end(self):
        from fastapi.testclient import TestClient
        from app import main
        main.llm = FakeLLM(replies=[json.dumps({"final": "Rent is $2,150 [C3]."})])
        c = TestClient(main.app)
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(len(c.get("/api/samples").json()), 2)
        r = c.post("/api/analyze", json={"text": LEASE}).json()
        self.assertEqual(len(r["clauses"]), 17)
        a = c.post("/api/ask", json={"doc_id": r["doc_id"], "question": "rent?"}).json()
        self.assertEqual(a["citations"], ["C3"])
        self.assertEqual(c.post("/api/ask", json={"doc_id": "nope", "question": "rent?"}).status_code, 404)
        self.assertEqual(c.post("/api/extract", content=b"plain text body").json()["text"], "plain text body")


if __name__ == "__main__":
    unittest.main()
