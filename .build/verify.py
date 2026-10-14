#!/usr/bin/env python3
"""Verify a lab notebook end-to-end with a real Jupyter kernel.

    .venv/bin/python .build/verify.py <course>/<notebook>.ipynb

Pass 1 (learner view): run as shipped — no cell may error, no exercise may pass, score == 0.
Pass 2 (solved view): replace each stub with its decoded solution — every exercise
must pass and score == max.
Both passes use a throwaway progress dir, so your real score is never touched.
"""

import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labkit import _decode  # noqa: E402


def run(nb, cwd, label):
    with tempfile.TemporaryDirectory() as prog:
        os.environ["LABKIT_PROGRESS_DIR"] = prog
        client = NotebookClient(nb, timeout=900, kernel_name="ai-engineer",
                                resources={"metadata": {"path": str(cwd)}}, allow_errors=True)
        client.execute()
        errors = []
        for i, c in enumerate(nb.cells):
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    errors.append((i, o["ename"], o["evalue"]))
        progress = {}
        for f in Path(prog).glob("*.json"):
            progress = json.loads(f.read_text())
    return errors, progress


def check_outputs(nb, ex_id):
    for c in nb.cells:
        if c.metadata.get("labkit", {}).get("role") == "tests" and c.metadata["labkit"]["exercise"] == ex_id:
            return "".join(o.get("text", "") for o in c.get("outputs", []))
    return ""


def main(path):
    path = Path(path).resolve()
    course = path.parent
    nb0 = nbformat.read(path, as_version=4)
    lab_id = next(c for c in nb0.cells if c.cell_type == "code" and "Lab(" in c.source).source.split('Lab("')[1].split('"')[0]
    sols = json.loads((course / ".solutions" / f"{lab_id}.json").read_text())["exercises"]
    ok = True

    print(f"== {path.name}: pass 1 (as shipped) ==")
    nb1 = copy.deepcopy(nb0)
    errors, prog = run(nb1, course, "shipped")
    for e in errors:
        print("  FAIL cell", e); ok = False
    passed = [k for k, v in prog.items() if v.get("passed_at")]
    if passed:
        print("  FAIL exercises pass with empty stubs:", passed); ok = False
    print(f"  cells errored: {len(errors)}; exercises passing unsolved: {passed or 'none'}")

    print(f"== {path.name}: pass 2 (stubs replaced by solutions) ==")
    nb2 = copy.deepcopy(nb0)
    for c in nb2.cells:
        m = c.metadata.get("labkit", {})
        if m.get("role") == "stub":
            c.source = _decode(sols[m["exercise"]]["solution"])
    errors, prog = run(nb2, course, "solved")
    for e in errors:
        print("  FAIL cell", e); ok = False
    for ex_id in sols:
        good = bool(prog.get(ex_id, {}).get("passed_at"))
        print(f"  {'PASS' if good else 'FAIL'} {ex_id} ({sols[ex_id]['points']} pts) {sols[ex_id]['title']}")
        if not good:
            ok = False
            print("    " + check_outputs(nb2, ex_id).replace("\n", "\n    "))
    print("== RESULT:", "PASS" if ok else "FAIL", "==")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([main(p) for p in sys.argv[1:]]) else 1)
