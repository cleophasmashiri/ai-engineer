"""labkit — graded exercises, hidden solutions and scoring for the lab notebooks.

Every notebook starts with:

    from labkit import Lab
    lab = Lab("lab_03_nlp_data_loader")

and then, per exercise:

    lab.check("ex1", test_a, test_b)   # run the provided tests, award points
    lab.solution_button("ex1")         # hidden solution (costs points if unsolved)

Scoring rules (per exercise worth P points):
  * pass the tests on your own                      -> +P
  * reveal the solution BEFORE you have passed      -> -P  (and passing afterwards earns nothing)
  * reveal the solution AFTER you have passed       ->  0  (free — compare approaches!)
Progress is saved to <course>/.progress/<lab_id>.json, so restarting the kernel
does not reset (or refund) anything. Use lab.reset(...) to start over.
"""

from __future__ import annotations

import base64
import json
import os
import time
import traceback
import zlib
from pathlib import Path

__all__ = ["Lab", "course_scoreboard"]


def _decode(blob: str) -> str:
    return zlib.decompress(base64.b64decode(blob)).decode("utf-8")


def encode(text: str) -> str:
    """Obfuscate a solution so it is not readable at a glance (used by the builder)."""
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _md(text: str) -> None:
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except Exception:
        print(text)


class Lab:
    def __init__(self, lab_id: str, base_dir: str | os.PathLike | None = None):
        self.lab_id = lab_id
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        sol_path = self.base_dir / ".solutions" / f"{lab_id}.json"
        if not sol_path.exists():
            raise FileNotFoundError(
                f"Can't find {sol_path}. Open the notebook from its own folder "
                f"(the kernel's working directory must be the course folder).")
        self.meta = json.loads(sol_path.read_text(encoding="utf-8"))
        self.exercises = self.meta["exercises"]  # ex_id -> {title, points, solution}
        prog_dir = Path(os.environ.get("LABKIT_PROGRESS_DIR") or self.base_dir / ".progress")
        prog_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = prog_dir / f"{lab_id}.json"
        self.state = self._load()
        total = sum(e["points"] for e in self.exercises.values())
        print(f"📒 {self.meta.get('title', lab_id)} — {len(self.exercises)} exercises, "
              f"{total} points available. Current score: {self.score()}/{total}")

    # ---------------------------------------------------------------- state
    def _load(self) -> dict:
        if self.progress_path.exists():
            try:
                return json.loads(self.progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    def _save(self) -> None:
        self.progress_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _ex(self, ex_id: str) -> dict:
        if ex_id not in self.exercises:
            raise KeyError(f"Unknown exercise {ex_id!r}. Known: {list(self.exercises)}")
        return self.state.setdefault(ex_id, {"attempts": 0, "passed_at": None, "revealed_at": None})

    def points_for(self, ex_id: str) -> int:
        s = self.state.get(ex_id) or {}
        p = self.exercises[ex_id]["points"]
        passed, revealed = s.get("passed_at"), s.get("revealed_at")
        if revealed and (not passed or revealed < passed):
            return -p
        return p if passed else 0

    def score(self) -> int:
        return sum(self.points_for(e) for e in self.exercises)

    def max_score(self) -> int:
        return sum(e["points"] for e in self.exercises.values())

    # ---------------------------------------------------------------- grading
    def check(self, ex_id: str, *tests) -> bool:
        """Run the provided tests for an exercise and update the score."""
        ex, s = self.exercises[ex_id], self._ex(ex_id)
        print(f"🧪 {ex_id}: {ex['title']}  ({ex['points']} pts)")
        passed = 0
        for t in tests:
            name = t.__name__.removeprefix("test_").replace("_", " ")
            try:
                t()
                print(f"   ✅ {name}")
                passed += 1
            except NotImplementedError:
                print(f"   ⬜ {name} — not implemented yet")
            except AssertionError as e:
                print(f"   ❌ {name} — {e or 'assertion failed'}")
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                where = f" (line {tb[-1].lineno} in {tb[-1].name})" if tb else ""
                print(f"   💥 {name} — {type(e).__name__}: {e}{where}")
        ok = passed == len(tests)
        s["attempts"] += 1
        if ok and not s["passed_at"]:
            s["passed_at"] = time.time()
        self._save()

        pts = self.points_for(ex_id)
        if ok:
            if s["revealed_at"] and s["revealed_at"] < s["passed_at"]:
                print(f"🟡 All {len(tests)} tests pass — but the solution was revealed first, "
                      f"so this exercise scores {pts}.")
            else:
                print(f"🎉 All {len(tests)} tests pass! +{pts} pts")
        else:
            print(f"   {passed}/{len(tests)} tests passing — keep going.")
        print(f"   Lab score: {self.score()}/{self.max_score()}")
        return ok

    # ---------------------------------------------------------------- solutions
    def reveal(self, ex_id: str, confirm: bool = False) -> None:
        """Show the hidden solution. Costs the exercise's points if not yet passed.

        Pass confirm=True to accept the penalty (the button does this for you).
        """
        ex, s = self.exercises[ex_id], self._ex(ex_id)
        free = bool(s["passed_at"]) or bool(s["revealed_at"])
        if not free and not confirm:
            print(f"🔒 Revealing {ex_id} now costs {ex['points']} pts. "
                  f"Call lab.reveal({ex_id!r}, confirm=True) to accept.")
            return
        if not s["passed_at"] and not s["revealed_at"]:
            s["revealed_at"] = time.time()
            self._save()
            print(f"🔓 Solution revealed: {ex_id} now scores -{ex['points']} "
                  f"(lab score {self.score()}/{self.max_score()})")
        _md(f"**Reference solution — {ex['title']}**\n\n```python\n"
            f"{_decode(ex['solution']).strip()}\n```")

    def solution_button(self, ex_id: str) -> None:
        """Render a two-click 'Show solution' button (falls back to text instructions)."""
        ex, s = self.exercises[ex_id], self._ex(ex_id)
        try:
            import ipywidgets as W
            from IPython.display import display
        except ImportError:
            W = None
        if W is None or not _in_notebook():
            print(f"💡 Stuck? lab.reveal({ex_id!r}, confirm=True) shows the solution "
                  f"(-{ex['points']} pts unless you've already passed).")
            return

        def label():
            if s["passed_at"]:
                return "👀 Compare with reference solution (free)"
            if s["revealed_at"]:
                return "👀 Show solution again"
            return f"🔒 Show solution (costs {ex['points']} pts)"

        btn = W.Button(description=label(), layout=W.Layout(width="auto"))
        out = W.Output()
        armed = {"v": False}

        def on_click(_):
            free = bool(s["passed_at"]) or bool(s["revealed_at"])
            if not free and not armed["v"]:
                armed["v"] = True
                btn.description = f"⚠️ Really? Click again to lose {ex['points']} pts"
                btn.button_style = "warning"
                return
            with out:
                out.clear_output()
                self.reveal(ex_id, confirm=True)
            btn.description, btn.button_style = label(), ""
            armed["v"] = False

        btn.on_click(on_click)
        display(W.VBox([btn, out]))

    # ---------------------------------------------------------------- reporting
    def scoreboard(self) -> None:
        rows = ["| Exercise | Points | Status | Score |", "|---|---:|---|---:|"]
        for ex_id, ex in self.exercises.items():
            s = self.state.get(ex_id) or {}
            pts = self.points_for(ex_id)
            if pts < 0:
                status = "🔓 revealed before solving"
            elif s.get("passed_at"):
                status = "✅ solved"
            elif s.get("attempts"):
                status = f"🔁 in progress ({s['attempts']} attempts)"
            else:
                status = "⬜ not started"
            rows.append(f"| {ex_id} — {ex['title']} | {ex['points']} | {status} | {pts:+d} |")
        rows.append(f"| **Total** | **{self.max_score()}** | | **{self.score():+d}** |")
        _md("\n".join(rows))

    def reset(self, ex_id: str | None = None, confirm: bool = False) -> None:
        """Wipe progress for one exercise (or the whole lab with ex_id=None)."""
        if not confirm:
            print("Pass confirm=True to really reset progress.")
            return
        if ex_id:
            self.state.pop(ex_id, None)
        else:
            self.state = {}
        self._save()
        print(f"Progress reset. Lab score: {self.score()}/{self.max_score()}")


def course_scoreboard(course_dir: str | os.PathLike = ".") -> None:
    """Summarise every lab in a course folder."""
    course = Path(course_dir)
    rows = ["| Lab | Score | Max |", "|---|---:|---:|"]
    tot = mx = 0
    for sol in sorted((course / ".solutions").glob("*.json")):
        lab_id = sol.stem
        meta = json.loads(sol.read_text(encoding="utf-8"))
        prog = course / ".progress" / f"{lab_id}.json"
        state = json.loads(prog.read_text()) if prog.exists() else {}
        lab = Lab.__new__(Lab)
        lab.exercises, lab.state = meta["exercises"], state
        rows.append(f"| {meta.get('title', lab_id)} | {lab.score():+d} | {lab.max_score()} |")
        tot += lab.score(); mx += lab.max_score()
    rows.append(f"| **Course total** | **{tot:+d}** | **{mx}** |")
    _md("\n".join(rows))
