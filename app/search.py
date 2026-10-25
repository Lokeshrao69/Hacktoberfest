"""Tiny dependency-free BM25 index over clauses (the agent's retrieval tool)."""
import math
import re
from collections import Counter

_STOP = set("""a an and are as at be by for from has have in is it its of on or that the
this to was were will with shall may any all such which who whom be been being not no
""".split())


def tokenize(text: str):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in toks if t not in _STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d["title"] + " " + d["text"])) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if docs else 0
        df = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k=4):
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1)))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.docs[i], round(s, 3)) for s, i in scores[:k]]
