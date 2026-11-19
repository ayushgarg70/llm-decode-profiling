"""Every image and relative link in the write-ups must point at a file that exists (moving a doc breaks them)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", "traces", "reports", "logs", ".pytest_cache"}
DOCS = sorted(p for p in ROOT.rglob("*.md") if not SKIP & set(p.relative_to(ROOT).parts))
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    missing = []
    text = re.sub(r"``.*?``", "", doc.read_text())  # examples inside code spans aren't links
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:", "<")):
            continue
        path = target.split("#")[0]
        if path and not (doc.parent / path).exists():
            missing.append(target)
    assert not missing, f"{doc.name}: broken links {missing}"
