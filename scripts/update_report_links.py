"""Keep code links in Markdown notes pointing at the right lines.

A link whose text is a backticked Python name and whose target is a .py file, e.g.
    [`KVCache.update`](../decode/model.py#L120-L135)      or      [`floor_ms`](../analysis/predict.py)
is rewritten to the current line range of that definition (function, class, method, or top-level
assignment). GitHub opens relative links like these at the highlighted lines.

    uv run python scripts/update_report_links.py            # rewrite notes/REPORT.md in place
    uv run python scripts/update_report_links.py --check    # exit 1 if any link is stale or broken
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

LINK = re.compile(r"\[`(?P<name>[A-Za-z_][\w.]*)`\]\((?P<path>[^)#\s]+\.py)(?P<anchor>#L\d+(?:-L\d+)?)?\)")


def find(tree: ast.Module, qualname: str) -> ast.AST | None:
    nodes: list[ast.AST] = list(tree.body)
    found = None
    for part in qualname.split("."):
        found = None
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == part:
                found = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == part for t in targets):
                    found = node
            if found:
                break
        if found is None:
            return None
        nodes = list(getattr(found, "body", []))
    return found


def update(md_path: Path, check: bool) -> int:
    text = md_path.read_text()
    problems, trees = [], {}

    def repl(m: re.Match) -> str:
        target = (md_path.parent / m["path"]).resolve()
        if not target.exists():
            problems.append(f"missing file: {m['path']}")
            return m[0]
        tree = trees.setdefault(target, ast.parse(target.read_text()))
        node = find(tree, m["name"])
        if node is None:
            problems.append(f"`{m['name']}` not found in {m['path']}")
            return m[0]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        anchor = f"#L{start}-L{node.end_lineno}"
        if m["anchor"] != anchor:
            problems.append(f"stale: `{m['name']}` {m['anchor'] or '(no lines)'} -> {anchor}")
        return f"[`{m['name']}`]({m['path']}{anchor})"

    new = LINK.sub(repl, text)
    for p in problems:
        print(p)
    if check:
        return 1 if problems else 0
    if new != text:
        md_path.write_text(new)
        print(f"updated {md_path}")
    return 1 if any(not p.startswith("stale") for p in problems) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["notes/REPORT.md"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sys.exit(max(update(Path(f), a.check) for f in a.files))
