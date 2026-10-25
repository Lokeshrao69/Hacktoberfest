"""Split a contract into addressable clauses (C1, C2, ...)."""
import re

_NUMBERED = re.compile(
    r"^\s*(\d+(\.\d+)*[.)]\s+\S"           # 1. Title / 3) Title
    r"|\d+(\.\d+)+\s+\S"                    # 1.2 Title
    r"|(section|article|clause)\s+[\dIVXLC]+)",  # Section 4 / Article IV
    re.I,
)
_ALLCAPS = re.compile(r"^[A-Z][A-Z0-9 ,&/\-]{3,60}:?$")
MAX_CLAUSE_CHARS = 1800


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _ALLCAPS.match(s) and len(s.split()) <= 8:
        return True
    return bool(_NUMBERED.match(line))


def _split_long(text: str):
    if len(text) <= MAX_CLAUSE_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+", text)
    parts, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > MAX_CLAUSE_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += s + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def segment(text: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    numbered = sum(1 for l in lines if _is_heading(l))

    blocks = []
    if numbered >= 3:
        cur = []
        for line in lines:
            if _is_heading(line) and cur and any(c.strip() for c in cur):
                blocks.append("\n".join(cur))
                cur = []
            cur.append(line)
        if cur:
            blocks.append("\n".join(cur))
    else:
        blocks = re.split(r"\n\s*\n", text)

    # Merge orphan headings ("12. TERMINATION") into the following block.
    merged, carry = [], ""
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(b) < 50 and "\n" not in b and not b.endswith("."):
            carry = (carry + "\n" + b).strip()
            continue
        merged.append((carry + "\n" + b).strip() if carry else b)
        carry = ""
    if carry:
        merged.append(carry)

    clauses = []
    for b in merged:
        for part in _split_long(b):
            first = part.split("\n", 1)[0].strip()
            title = first if len(first) <= 80 else first[:77] + "..."
            clauses.append({"id": f"C{len(clauses) + 1}", "title": title, "text": part})
    return clauses
