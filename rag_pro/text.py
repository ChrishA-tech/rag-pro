from __future__ import annotations

import re
from typing import List

_STOP = set(
    "a an the of to in on for and or is are was were be been being it its this that these those with as by at "
    "from how what which who whom why when where do does did can could should would will you your i we they he "
    "she not no than then so if into about over under between via using use used there their them our out up "
    "each any some more most also may might must have has had get gets".split()
)
_TOK = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


def stem(t: str) -> str:
    """Tiny suffix stripper - good enough for lexical matching, no external deps."""
    if len(t) <= 3 or t.isdigit():
        return t
    for _ in range(2):
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("ing") and len(t) > 5:
            t = t[:-3]
        elif t.endswith("ed") and len(t) > 4:
            t = t[:-2]
        elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            t = t[:-1]
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def tokenize(text: str, remove_stop: bool = True) -> List[str]:
    toks = _TOK.findall(text.lower())
    return [stem(t) for t in toks if not (remove_stop and t in _STOP)]


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in _SENT.split(text) if s]
