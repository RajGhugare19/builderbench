"""Regression tests for F-09 (run output isolation).

These tests exercise ``output_paths`` — the stdlib-only helper that both
``run.py`` and ``submit.py`` now depend on — so they run without installing
mujoco, mediapy, agents, numpy, openai, or simple-parsing.

They cover:
    * timestamp format has microsecond precision and is round-trippable;
    * ``build_run_dirname`` matches the README layout and validates inputs;
    * ``parse_run_dirname`` recognises both the new timestamped directory
      and the pre-fix legacy directory, and rejects look-alikes;
    * ``select_latest_per_seed`` collapses reruns of the same seed instead
      of letting ``submit.py`` silently double-count them;
    * ``Path.mkdir(exist_ok=False, parents=True)`` (the exact call in
      ``run.py``) creates distinct directories for distinct timestamps and
      raises ``FileExistsError`` when a same-named directory already exists;
    * the ``eval_summary.jsonl`` files written to two runs do not overlap
      and can be re-read with the same ``json.loads`` shape submit.py
      uses.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import output_paths as op  # noqa: E402


TS_A = "2026-11-06_05-30-15-000001"
TS_B = "2026-11-06_05-30-15-000002"
TS_LATE = "2026-11-06_05-30-15-654321"


class FormatTimestampTests(unittest.TestCase):
    def test_matches_microsecond_pattern(self):
        got = op.format_timestamp(datetime(2026, 11, 6, 5, 30, 15, 123456))
        self.assertEqual(got, "2026-11-06_05-30-15-123456")

    def test_zero_pads_microseconds(self):
        got = op.format_timestamp(datetime(2026, 1, 2, 3, 4, 5, 6))
        self.assertEqual(got, "2026-01-02_03-04-05-000006")

    def test_two_distinct_us_datetimes_give_distinct_strings(self):
        a = op.format_timestamp(datetime(2026, 11, 6, 5, 30, 15, 1))
        b = op.format_timestamp(datetime(2026, 11, 6, 5, 30, 15, 2))
        self.assertNotEqual(a, b, "same-second launches must be distinguishable")

    def test_rejects_non_datetime(self):
        with self.assertRaises(TypeError):
            op.format_timestamp("2026-11-06_05-30-15-123456")


class BuildRunDirnameTests(unittest.TestCase):
    def test_returns_expected_layout(self):
        got = op.build_run_dirname("cube-1-task-1", 0, TS_A)
        self.assertEqual(got, f"cube-1-task-1-seed-0-timestamp-{TS_A}")

    def test_rejects_empty_level(self):
        with self.assertRaises(ValueError):
            op.build_run_dirname("", 0, TS_A)

    def test_rejects_negative_seed(self):
        with self.assertRaises(ValueError):
            op.build_run_dirname("cube-1-task-1", -1, TS_A)

    def test_rejects_bool_seed(self):
        # bool is a subclass of int; the guard must reject True/False.
        with self.assertRaises(ValueError):
            op.build_run_dirname("cube-1-task-1", True, TS_A)

    def test_rejects_second_granularity_timestamp(self):
        # The bug this fix removes: second-precision suffixes collide on
        # back-to-back launches.  Guard the format.
        with self.assertRaises(ValueError):
            op.build_run_dirname("cube-1-task-1", 0, "2026-11-06_05-30-15")

    def test_rejects_partial_microseconds(self):
        with self.assertRaises(ValueError):
            op.build_run_dirname("cube-1-task-1", 0, "2026-11-06_05-30-15-1234")


class ParseRunDirnameTests(unittest.TestCase):
    def test_new_layout(self):
        parsed = op.parse_run_dirname(f"cube-1-task-1-seed-0-timestamp-{TS_A}")
        self.assertEqual(parsed, ("cube-1-task-1", 0, TS_A))

    def test_legacy_layout(self):
        parsed = op.parse_run_dirname("cube-1-task-1-seed-7")
        self.assertEqual(parsed, ("cube-1-task-1", 7, None))

    def test_unrelated_name_returns_none(self):
        self.assertIsNone(op.parse_run_dirname("scratch-note"))

    def test_lookalike_suffix_is_not_silently_accepted(self):
        # A directory whose suffix *starts* like a timestamp but does not
        # finish it must not be mis-parsed as seed=0.
        self.assertIsNone(
            op.parse_run_dirname("cube-1-task-1-seed-0-timestamp-2026-11-06_05-30-15-1234")
        )

    def test_roundtrip_matches_build(self):
        name = op.build_run_dirname("cube-9-task-3", 42, TS_LATE)
        self.assertEqual(op.parse_run_dirname(name), ("cube-9-task-3", 42, TS_LATE))


class SelectLatestPerSeedTests(unittest.TestCase):
    """These are the semantics submit.py relies on so an old
    ``-seed-0`` directory and a new ``-seed-0-timestamp-<...>`` sibling are
    not double-counted as two independent runs of the same seed."""

    def _paths(self, names):
        return [Path("outputs/naive/qwen") / n for n in names]

    def test_timestamped_beats_legacy(self):
        winners = op.select_latest_per_seed(
            self._paths(["cube-1-task-1-seed-0", f"cube-1-task-1-seed-0-timestamp-{TS_A}"])
        )
        self.assertEqual([w.name for w in winners], [f"cube-1-task-1-seed-0-timestamp-{TS_A}"])

    def test_only_legacy_used_when_no_timestamped(self):
        winners = op.select_latest_per_seed(self._paths(["cube-1-task-1-seed-0"]))
        self.assertEqual([w.name for w in winners], ["cube-1-task-1-seed-0"])

    def test_newer_timestamp_wins_same_seed(self):
        winners = op.select_latest_per_seed(
            self._paths([
                f"cube-1-task-1-seed-0-timestamp-{TS_A}",
                f"cube-1-task-1-seed-0-timestamp-{TS_B}",
            ])
        )
        self.assertEqual([w.name for w in winners], [f"cube-1-task-1-seed-0-timestamp-{TS_B}"])

    def test_multiple_seeds_preserved(self):
        winners = op.select_latest_per_seed(
            self._paths([
                f"cube-1-task-1-seed-0-timestamp-{TS_A}",
                f"cube-1-task-1-seed-1-timestamp-{TS_B}",
                f"cube-1-task-1-seed-2-timestamp-{TS_LATE}",
            ])
        )
        self.assertEqual(
            [w.name for w in winners],
            [
                f"cube-1-task-1-seed-0-timestamp-{TS_A}",
                f"cube-1-task-1-seed-1-timestamp-{TS_B}",
                f"cube-1-task-1-seed-2-timestamp-{TS_LATE}",
            ],
        )

    def test_ignores_unrelated_dirs(self):
        winners = op.select_latest_per_seed(
            self._paths([
                f"cube-1-task-1-seed-0-timestamp-{TS_A}",
                "scratch-note",
            ])
        )
        self.assertEqual([w.name for w in winners], [f"cube-1-task-1-seed-0-timestamp-{TS_A}"])

    def test_seed_ordering_is_stable(self):
        # Ordering by seed lets submit.py iterate deterministically; do not
        # leak the glob order.
        paths = self._paths([
            f"cube-1-task-1-seed-2-timestamp-{TS_A}",
            f"cube-1-task-1-seed-10-timestamp-{TS_A}",
            f"cube-1-task-1-seed-1-timestamp-{TS_A}",
        ])
        winners = op.select_latest_per_seed(paths)
        self.assertEqual([w.name for w in winners], [
            f"cube-1-task-1-seed-1-timestamp-{TS_A}",
            f"cube-1-task-1-seed-2-timestamp-{TS_A}",
            f"cube-1-task-1-seed-10-timestamp-{TS_A}",
        ])


class RunDirectoryCreationTests(unittest.TestCase):
    """Exercise the exact ``Path.mkdir`` call that ``run.py`` now uses."""

    def _create(self, root, level_id, seed, ts):
        leaf = op.build_run_dirname(level_id, seed, ts)
        path = Path(root) / "naive" / "qwen" / leaf
        path.mkdir(exist_ok=False, parents=True)
        return path

    def test_two_runs_same_config_get_distinct_dirs(self):
        with tempfile.TemporaryDirectory() as root:
            a = self._create(root, "cube-1-task-1", 0, TS_A)
            b = self._create(root, "cube-1-task-1", 0, TS_B)
            self.assertNotEqual(a, b)
            self.assertTrue(a.is_dir() and b.is_dir())

    def test_same_timestamp_collision_raises(self):
        with tempfile.TemporaryDirectory() as root:
            self._create(root, "cube-1-task-1", 0, TS_A)
            with self.assertRaises(FileExistsError):
                self._create(root, "cube-1-task-1", 0, TS_A)

    def test_written_eval_summaries_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as root:
            a = self._create(root, "cube-1-task-1", 0, TS_A)
            b = self._create(root, "cube-1-task-1", 0, TS_B)
            (a / "eval_summary.jsonl").write_text(
                json.dumps({"episode_idx": 0, "success": False}) + "\n", encoding="utf-8"
            )
            (b / "eval_summary.jsonl").write_text(
                json.dumps({"episode_idx": 0, "success": True}) + "\n", encoding="utf-8"
            )
            a_records = json.loads((a / "eval_summary.jsonl").read_text(encoding="utf-8").strip())
            b_records = json.loads((b / "eval_summary.jsonl").read_text(encoding="utf-8").strip())
            self.assertFalse(a_records["success"])
            self.assertTrue(b_records["success"])

    def test_submit_can_discover_new_and_legacy_siblings(self):
        # Reproduces the mixed layout reviewers called out: a legacy dir
        # and two timestamped reruns of the same seed coexist under
        # outputs/<agent>/<model>/.  select_latest_per_seed still returns
        # one winner per seed chosen by directory name; submit.py's own
        # fallback logic (test_submission_selection_safety) is responsible
        # for probing the winner's usability.
        with tempfile.TemporaryDirectory() as root:
            root_p = Path(root)
            (root_p / "cube-1-task-1-seed-0").mkdir()
            (root_p / f"cube-1-task-1-seed-0-timestamp-{TS_A}").mkdir()
            (root_p / f"cube-1-task-1-seed-0-timestamp-{TS_B}").mkdir()
            candidates = (d for d in root_p.glob("cube-1-task-1-seed-*/") if d.is_dir())
            winners = op.select_latest_per_seed(candidates)
            self.assertEqual([w.name for w in winners], [f"cube-1-task-1-seed-0-timestamp-{TS_B}"])


# ----------------------------------------------------------------------
# F-09 selection-safety: newer→older fallback inside submit.collect_task_runs
# ----------------------------------------------------------------------
#
# These tests import submit.py after stubbing ``simple_parsing`` (its
# ArgumentParser is only used under submit.py's ``if __name__=="__main__"``
# branch, so the stub never executes).  numpy and builderbench.create_task_data
# are already installed and have no import-time file writes.
#
# They also run the *negative control*: the pre-fix pipeline (name-only
# winner + no usability probe) is reconstructed in-line and asserted to
# lose data on S1 / S2 / S7, so we can prove the new fallback is
# load-bearing.

import os
import types
import contextlib
import io


def _install_simple_parsing_stub():
    if "simple_parsing" in sys.modules and hasattr(sys.modules["simple_parsing"], "ArgumentParser"):
        return
    stub = types.ModuleType("simple_parsing")

    class _ArgumentParserStub:
        def __init__(self, *a, **k):
            pass

        def add_arguments(self, *a, **k):
            pass

        def parse_args(self, *a, **k):
            raise SystemExit("submit.main() invoked outside __main__")

    stub.ArgumentParser = _ArgumentParserStub
    stub.field = lambda *a, **k: None
    sys.modules["simple_parsing"] = stub


_install_simple_parsing_stub()
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import submit as _submit  # noqa: E402


TS_OLD = "2026-11-06_05-30-15-000000"
TS_NEW = "2026-11-06_05-30-16-000000"
TS_LATEST = "2026-11-06_05-30-17-000000"
LEVEL = "cube-1-task-1"


def _write_metadata(p: Path, *, commit="commit-abc", model="model-x", ts=TS_OLD):
    meta = {
        "run_timestamp": ts,
        "builderbench_git_commit": commit,
        "task_version": {"level_id": LEVEL, "task_file_sha256": "sha-1"},
        "model": {"model_id": model, "client_name": "openai"},
        "dependency_versions": {"numpy": "2.0"},
    }
    (p / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_episode(p: Path, *, episode_idx=0, success=True):
    record = {
        "episode_idx": episode_idx,
        "success": success,
        "easy_success": success,
        "end_effector_success": success,
        "return": 1.0,
        "steps": 100,
        "num_actions": 5,
        "tokens_in": 100,
        "tokens_out": 40,
        "episode_time": "0:00:10",
        "per_cube_success": [success],
        "per_cube_easy_success": [success],
        "per_cube_mask": [1],
    }
    (p / "eval_summary.jsonl").write_text(
        json.dumps(record) + "\n\n", encoding="utf-8"
    )


def _write_full_run(p: Path, **kw):
    p.mkdir(parents=True, exist_ok=True)
    _write_metadata(p, ts=kw.pop("ts", TS_OLD), **kw)
    (p / "run_config.json").write_text("{}", encoding="utf-8")
    _write_episode(p, **{k: v for k, v in kw.items() if k in ("episode_idx", "success")})


def _write_incomplete_run(p: Path, *, ts=TS_NEW, commit="commit-abc", model="model-x"):
    """Simulates the shape run.py leaves behind when a rerun is killed
    after opening eval_summary.jsonl (mode 'w') but before writing any
    episode: metadata and config present, eval_summary truncated empty."""
    p.mkdir(parents=True, exist_ok=True)
    _write_metadata(p, commit=commit, model=model, ts=ts)
    (p / "run_config.json").write_text("{}", encoding="utf-8")
    (p / "eval_summary.jsonl").write_text("", encoding="utf-8")


def _write_corrupt_summary(p: Path, *, ts=TS_NEW):
    p.mkdir(parents=True, exist_ok=True)
    _write_metadata(p, ts=ts)
    (p / "eval_summary.jsonl").write_text("{not valid json}\n\n", encoding="utf-8")


def _write_empty_run_dir(p: Path):
    """A directory that was created by run.py's mkdir but killed before
    any JSON file was written."""
    p.mkdir(parents=True, exist_ok=True)


_TEMPDIRS: list[str] = []


def _run_layout(builder):
    """Give each scenario its own tempdir that outlives the ``return``.
    ``tempfile.TemporaryDirectory`` cleans up on ``__exit__``; ``mkdtemp``
    does not, so the tree survives until :func:`tearDownModule` runs."""
    d = tempfile.mkdtemp(prefix="f09b_")
    _TEMPDIRS.append(d)
    root = Path(d)
    builder(root)
    return root


def tearDownModule():
    for d in _TEMPDIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMPDIRS.clear()


def _main_config(results_root: Path, agent="a", model="m", level=LEVEL):
    return _submit.Args(
        results_dir=str(results_root),
        level_id=level,
        model_id=model,
        agent_name=agent,
        website_url="N/A",
    )


def _nested(base: Path, agent="a", model="m") -> Path:
    """submit.main composes ``results_dir/agent/model`` — helper that
    builds the same layout from a scenario's ``results_dir`` root."""
    out = base / agent / model
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextlib.contextmanager
def _chdir(target: Path):
    previous = os.getcwd()
    os.chdir(str(target))
    try:
        yield
    finally:
        os.chdir(previous)


class SubmissionSelectionFallbackTests(unittest.TestCase):
    """S1..S10 forward tests plus the negative control demonstrating the
    pre-fix pipeline's silent-drop behaviour."""

    # -------- S1: legacy complete + timestamped incomplete --------

    def test_S1_falls_back_to_legacy_when_newer_is_incomplete(self):
        def build(root):
            _write_full_run(root / "cube-1-task-1-seed-0")
            _write_incomplete_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_NEW}")

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(len(runs), 1)
        chosen = Path(runs[0]["path"])
        self.assertEqual(chosen.name, "cube-1-task-1-seed-0")
        self.assertEqual(len(runs[0]["skipped"]), 1)
        self.assertIn("empty", runs[0]["skipped"][0]["reason"])
        self.assertIn(TS_NEW, runs[0]["skipped"][0]["path"])

    def test_S1_negative_control_pre_fix_pipeline_would_have_silently_dropped_data(self):
        """Pre-fix: name-based winner selection + load without probe →
        compute_task_metrics reports ``num_seeds=0`` and ``main`` exits 0
        without any leaderboard being written."""
        def build(root):
            _write_full_run(root / "cube-1-task-1-seed-0")
            _write_incomplete_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_NEW}")

        root = _run_layout(build)
        # Reproduce the pre-fix winner: name-only tiebreak.
        candidates = (d for d in root.glob(f"{LEVEL}-seed-*/") if d.is_dir())
        pre_fix_winner = op.select_latest_per_seed(candidates)[0]
        self.assertIn("timestamp", pre_fix_winner.name)
        # Pre-fix loaded eval_summary without any probe.
        pre_fix_episodes = _submit.load_eval_summary(
            pre_fix_winner / "eval_summary.jsonl"
        )
        self.assertEqual(pre_fix_episodes, [])
        pre_fix_metrics = _submit.compute_task_metrics([
            {"seed": 0, "eval_summary": pre_fix_episodes}
        ])
        self.assertEqual(pre_fix_metrics, {"num_seeds": 0})
        # The new probe rejects the same candidate, which is exactly the
        # signal the pre-fix pipeline lacked.
        usable, reason, _ = _submit._load_run_record(pre_fix_winner, LEVEL)
        self.assertFalse(usable)
        self.assertIn("empty", reason)

    # -------- S2: older timestamped complete + newer timestamped incomplete --------

    def test_S2_falls_back_to_older_timestamped(self):
        def build(root):
            _write_full_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_OLD}")
            _write_incomplete_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_NEW}")

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(len(runs), 1)
        chosen = Path(runs[0]["path"])
        self.assertEqual(chosen.name, f"cube-1-task-1-seed-0-timestamp-{TS_OLD}")
        self.assertEqual(len(runs[0]["skipped"]), 1)
        self.assertIn(TS_NEW, runs[0]["skipped"][0]["path"])
        self.assertIn("empty", runs[0]["skipped"][0]["reason"])

    def test_S2_negative_control_pre_fix_pipeline_silently_returns_no_winner(self):
        def build(root):
            _write_full_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_OLD}")
            _write_incomplete_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_NEW}")

        root = _run_layout(build)
        candidates = (d for d in root.glob(f"{LEVEL}-seed-*/") if d.is_dir())
        winner = op.select_latest_per_seed(candidates)[0]
        self.assertIn(TS_NEW, winner.name)
        episodes = _submit.load_eval_summary(winner / "eval_summary.jsonl")
        self.assertEqual(episodes, [])

    # -------- S3: two valid timestamped → newer wins with no skip --------

    def test_S3_picks_newer_and_records_no_skip(self):
        def build(root):
            _write_full_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
            _write_full_run(root / f"cube-1-task-1-seed-0-timestamp-{TS_NEW}", ts=TS_NEW)

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(len(runs), 1)
        self.assertEqual(Path(runs[0]["path"]).name, f"cube-1-task-1-seed-0-timestamp-{TS_NEW}")
        self.assertEqual(runs[0]["skipped"], [])

    # -------- S4: only legacy valid --------

    def test_S4_only_legacy_is_selected(self):
        def build(root):
            _write_full_run(root / "cube-1-task-1-seed-0")

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(len(runs), 1)
        self.assertEqual(Path(runs[0]["path"]).name, "cube-1-task-1-seed-0")
        self.assertEqual(runs[0]["skipped"], [])
        self.assertEqual(len(runs[0]["eval_summary"]), 1)

    # -------- S5: multiple seeds all valid --------

    def test_S5_multiple_seeds_aggregate(self):
        def build(root):
            for seed, ts in [(0, TS_OLD), (1, TS_NEW), (2, TS_LATEST)]:
                _write_full_run(root / f"{LEVEL}-seed-{seed}-timestamp-{ts}", ts=ts)

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual([r["seed"] for r in runs], [0, 1, 2])
        metrics = _submit.compute_task_metrics(runs)
        self.assertEqual(metrics["num_seeds"], 3)

    # -------- S6: all candidates invalid → unusable + main returns 2 --------

    def test_S6_all_candidates_invalid_marks_seed_unusable(self):
        def build(root):
            _write_empty_run_dir(root / f"{LEVEL}-seed-0-timestamp-{TS_OLD}")
            _write_empty_run_dir(root / f"{LEVEL}-seed-0-timestamp-{TS_NEW}")

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(runs, [])
        self.assertEqual(len(unusable), 1)
        entry = unusable[0]
        self.assertEqual(entry["seed"], 0)
        self.assertEqual(len(entry["candidates"]), 2)
        for item in entry["candidates"]:
            self.assertIn("run_metadata.json missing", item["reason"])

    def test_S6_main_returns_nonzero_when_no_usable_candidate(self):
        root = _run_layout(lambda r: _write_empty_run_dir(
            _nested(r) / f"{LEVEL}-seed-0-timestamp-{TS_NEW}"
        ))
        with _chdir(root), contextlib.redirect_stderr(io.StringIO()) as err:
            code = _submit.main(_main_config(root))
        self.assertEqual(code, 2)
        diag = err.getvalue()
        self.assertIn("seed=0", diag)
        self.assertIn("run_metadata.json missing", diag)
        self.assertIn("refusing to submit", diag)

    # -------- S7: one seed entirely invalid → cannot hide --------

    def test_S7_mixed_valid_and_invalid_seed_flags_unusable(self):
        def build(root):
            _write_full_run(root / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
            _write_empty_run_dir(root / f"{LEVEL}-seed-1-timestamp-{TS_NEW}")

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual([r["seed"] for r in runs], [0])
        self.assertEqual([e["seed"] for e in unusable], [1])

    def test_S7_main_refuses_to_silently_short_aggregate(self):
        def build(root):
            _write_full_run(root / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
            _write_empty_run_dir(root / f"{LEVEL}-seed-1-timestamp-{TS_NEW}")

        root = _run_layout(build)
        nested = _nested(root)
        # Rebuild inside the agent/model path main expects.
        with tempfile.TemporaryDirectory() as d2:
            root2 = Path(d2)
            nested2 = _nested(root2)
            _write_full_run(nested2 / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
            _write_empty_run_dir(nested2 / f"{LEVEL}-seed-1-timestamp-{TS_NEW}")
            with _chdir(root2), contextlib.redirect_stderr(io.StringIO()) as err:
                code = _submit.main(_main_config(root2))
        self.assertEqual(code, 2)
        self.assertIn("seed=1", err.getvalue())

    def test_S7_negative_control_pre_fix_silently_undercounted_seeds(self):
        """Pre-fix: seed 1's empty candidate was added as a run with
        eval_summary=[], compute_task_metrics then dropped it via
        ``if not episodes: continue`` and produced a leaderboard with
        num_seeds=1 — no visible signal that seed 1 was discovered but
        unusable."""
        def build(root):
            _write_full_run(root / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
            _write_empty_run_dir(root / f"{LEVEL}-seed-1-timestamp-{TS_NEW}")

        root = _run_layout(build)
        candidates = (d for d in root.glob(f"{LEVEL}-seed-*/") if d.is_dir())
        pre_fix_winners = op.select_latest_per_seed(candidates)
        self.assertEqual(len(pre_fix_winners), 2)
        # Pre-fix loaded them without a probe: seed 1 has an empty eval_summary.
        pre_fix_runs = []
        for winner in pre_fix_winners:
            pre_fix_runs.append({
                "seed": op.parse_run_dirname(winner.name)[1],
                "eval_summary": _submit.load_eval_summary(winner / "eval_summary.jsonl"),
            })
        metrics = _submit.compute_task_metrics(pre_fix_runs)
        # num_seeds silently dropped from 2 to 0 (both records were
        # evaluated with an empty episodes list because the pre-fix
        # code did not read run_metadata either; even the seed-0 winner
        # here is name-sorted so this shows the undercount path).
        self.assertLess(metrics.get("num_seeds", 0), 2)
        # Contrast: the new probe flags seed 1 as unusable and preserves
        # seed 0's real record.
        new_runs, new_unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual([r["seed"] for r in new_runs], [0])
        self.assertEqual([e["seed"] for e in new_unusable], [1])

    # -------- S8: latest has corrupt JSON, older valid --------

    def test_S8_corrupt_json_triggers_controlled_fallback(self):
        def build(root):
            _write_full_run(root / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
            _write_corrupt_summary(root / f"{LEVEL}-seed-0-timestamp-{TS_NEW}", ts=TS_NEW)

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(Path(runs[0]["path"]).name, f"{LEVEL}-seed-0-timestamp-{TS_OLD}")
        self.assertEqual(len(runs[0]["skipped"]), 1)
        self.assertIn("not parseable", runs[0]["skipped"][0]["reason"])

    # -------- S9: partial-episode run is NOT distinguishable --------

    def test_S9_partial_run_is_usable_and_limitation_is_visible(self):
        """run.py writes eval_summary.jsonl incrementally, so a run that
        produced 1 of 3 planned episodes and then crashed looks identical
        to a completed 1-episode run under the current schema.  The probe
        intentionally treats both as usable; there is no ``is_complete``
        or ``num_planned_episodes`` field the record could carry."""
        def build(root):
            p = root / f"{LEVEL}-seed-0-timestamp-{TS_NEW}"
            p.mkdir()
            _write_metadata(p, ts=TS_NEW)
            (p / "run_config.json").write_text("{}", encoding="utf-8")
            # Only episode 0 written; planner asked for 3.
            (p / "eval_summary.jsonl").write_text(
                json.dumps({"episode_idx": 0, "success": False}) + "\n\n",
                encoding="utf-8",
            )

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(len(runs), 1)
        record = runs[0]
        self.assertNotIn("is_complete", record)
        self.assertNotIn("num_planned_episodes", record)
        self.assertEqual(len(record["eval_summary"]), 1)

    # -------- S10: mixed git commits → consistency check still fires --------

    def test_S10_mixed_commits_still_rejected_by_verify_consistency(self):
        def build(root):
            _write_full_run(root / f"{LEVEL}-seed-0-timestamp-{TS_OLD}",
                            ts=TS_OLD, commit="commit-aaa")
            _write_full_run(root / f"{LEVEL}-seed-1-timestamp-{TS_NEW}",
                            ts=TS_NEW, commit="commit-bbb")

        root = _run_layout(build)
        runs, unusable = _submit.collect_task_runs(root, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(len(runs), 2)
        with self.assertRaises(ValueError) as ctx:
            _submit.verify_consistency(runs)
        self.assertIn("commit", str(ctx.exception).lower())

    def test_S10_main_returns_1_on_commit_mismatch(self):
        root = _run_layout(lambda r: None)
        nested = _nested(root)
        _write_full_run(nested / f"{LEVEL}-seed-0-timestamp-{TS_OLD}",
                        ts=TS_OLD, commit="commit-aaa")
        _write_full_run(nested / f"{LEVEL}-seed-1-timestamp-{TS_NEW}",
                        ts=TS_NEW, commit="commit-bbb")
        with _chdir(root), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = _submit.main(_main_config(root))
        self.assertEqual(code, 1)


class SubmissionDiagnosticsTests(unittest.TestCase):
    """Verify main's stderr diagnostics carry the three pieces a reviewer
    asked for: seed, skipped path, reason."""

    def test_skip_diagnostics_include_seed_path_and_reason(self):
        root = _run_layout(lambda r: None)
        nested = _nested(root)
        _write_full_run(nested / "cube-1-task-1-seed-0")
        _write_incomplete_run(nested / f"cube-1-task-1-seed-0-timestamp-{TS_NEW}")
        with _chdir(root), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = _submit.main(_main_config(root))
        self.assertEqual(code, 0)
        diag = err.getvalue()
        self.assertIn("[submit-fallback] seed=0", diag)
        self.assertIn("cube-1-task-1-seed-0-timestamp-" + TS_NEW, diag)
        self.assertIn("reason: eval_summary.jsonl empty", diag)


class ShellScriptLevelIdTests(unittest.TestCase):
    """All three scripts/submit_*.sh must derive the original level_id
    from both legacy (``<level>-seed-N``) and timestamped
    (``<level>-seed-N-timestamp-...``) directory names.  The tests run the
    actual ``sed`` expression extracted from each shipped script, and
    check the script's ``find -name`` glob matches both shapes."""

    SCRIPTS = (
        "submit_claude_opus4.6.sh",
        "submit_gemini_flash3.1.sh",
        "submit_openai_gpt5.2.sh",
    )
    NAMES = {
        "cube-1-task-1-seed-0": "cube-1-task-1",
        "cube-1-task-1-seed-12": "cube-1-task-1",
        f"cube-1-task-1-seed-0-timestamp-{TS_OLD}": "cube-1-task-1",
        f"cube-9-task-3-seed-7-timestamp-{TS_LATEST}": "cube-9-task-3",
    }

    def _script_text(self, name):
        return (_REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")

    def test_find_glob_matches_both_layouts(self):
        import fnmatch
        for script in self.SCRIPTS:
            text = self._script_text(script)
            match = re.search(r"-name '([^']+)'", text)
            self.assertIsNotNone(match, script)
            pattern = match.group(1)
            for name in self.NAMES:
                self.assertTrue(fnmatch.fnmatch(name, pattern), (script, name))

    @unittest.skipUnless(shutil.which("sed"), "sed not available")
    def test_sed_expression_extracts_level_id_for_both_layouts(self):
        import subprocess
        for script in self.SCRIPTS:
            text = self._script_text(script)
            match = re.search(r"sed -E '([^']+)'", text)
            self.assertIsNotNone(match, script)
            expression = match.group(1)
            names = list(self.NAMES)
            result = subprocess.run(
                ["sed", "-E", expression],
                input="\n".join(names) + "\n",
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                [self.NAMES[n] for n in names],
                script,
            )


def _write_raw_summary_run(p: Path, payload: str, *, ts=TS_NEW):
    """Run dir with valid metadata/config but a caller-controlled raw
    eval_summary.jsonl payload (for schema / non-finite probes)."""
    p.mkdir(parents=True, exist_ok=True)
    _write_metadata(p, ts=ts)
    (p / "run_config.json").write_text("{}", encoding="utf-8")
    (p / "eval_summary.jsonl").write_text(payload, encoding="utf-8")


class EpisodeSchemaValidationTests(unittest.TestCase):
    """The usability probe must reject anything compute_task_metrics
    cannot consume, so no accepted record can crash aggregation."""

    def _probe(self, payload):
        root = _run_layout(lambda r: None)
        p = root / f"{LEVEL}-seed-0-timestamp-{TS_NEW}"
        _write_raw_summary_run(p, payload)
        return _submit._load_run_record(p, LEVEL)

    def test_non_object_episodes_rejected(self):
        for payload in ("null\n\n", "[1, 2]\n\n", "42\n\n", '"text"\n\n', "true\n\n"):
            ok, reason, record = self._probe(payload)
            self.assertFalse(ok, payload)
            self.assertIsNone(record)
            self.assertIn("not a JSON object", reason)

    def test_missing_consumed_fields_keep_default_behaviour(self):
        # Every field compute_task_metrics reads has a .get default; an
        # episode carrying none of them is a legal all-zero record.
        ok, reason, record = self._probe('{"episode_idx": 0}\n\n')
        self.assertTrue(ok, reason)
        metrics = _submit.compute_task_metrics([record])
        self.assertEqual(metrics["num_seeds"], 1)
        self.assertEqual(metrics["mean_top1_success_rate"], 0.0)
        self.assertEqual(metrics["mean_input_tokens_per_run"], 0.0)

    def test_wrong_field_types_rejected(self):
        cases = [
            ('{"per_cube_success": "True"}\n\n', "per_cube_success is not a list"),
            ('{"per_cube_success": [true, "x"]}\n\n', "not numeric"),
            ('{"per_cube_mask": 5}\n\n', "per_cube_mask is not a list"),
        ]
        for payload, fragment in cases:
            ok, reason, _ = self._probe(payload)
            self.assertFalse(ok, payload)
            self.assertIn(fragment, reason)

    def test_illegal_token_values_rejected_without_uncaught_exception(self):
        for payload in (
            '{"tokens_in": "abc"}\n\n',
            '{"tokens_out": "1.5"}\n\n',
            '{"tokens_in": {"nested": 1}}\n\n',
        ):
            ok, reason, _ = self._probe(payload)
            self.assertFalse(ok, payload)
            self.assertIn("integer token count", reason)

    def test_historical_token_defaults_still_accepted(self):
        for payload in (
            '{"tokens_in": null, "tokens_out": null}\n\n',
            '{"tokens_in": "12", "tokens_out": 3.9}\n\n',
        ):
            ok, reason, record = self._probe(payload)
            self.assertTrue(ok, (payload, reason))
            metrics = _submit.compute_task_metrics([record])
            self.assertEqual(metrics["num_seeds"], 1)

    def test_non_finite_values_rejected(self):
        cases = [
            '{"tokens_in": NaN}\n\n',
            '{"tokens_out": Infinity}\n\n',
            '{"tokens_in": -Infinity}\n\n',
            '{"per_cube_success": [NaN]}\n\n',
            '{"per_cube_success": [true, Infinity]}\n\n',
            '{"per_cube_success": [-Infinity]}\n\n',
        ]
        for payload in cases:
            ok, reason, _ = self._probe(payload)
            self.assertFalse(ok, payload)
            self.assertIn("not a finite number", reason)

    def test_json_numeric_overflow_to_infinity_rejected(self):
        # json.loads("1e999") yields float('inf') without raising.
        for payload in (
            '{"tokens_in": 1e999}\n\n',
            '{"per_cube_success": [1e999]}\n\n',
            '{"tokens_out": -1e999}\n\n',
        ):
            ok, reason, _ = self._probe(payload)
            self.assertFalse(ok, payload)
            self.assertIn("not a finite number", reason)

    def test_second_episode_invalid_rejects_whole_candidate(self):
        valid = json.dumps({"episode_idx": 0, "success": True})
        ok, reason, _ = self._probe(valid + "\n\n" + "null\n\n")
        self.assertFalse(ok)
        self.assertIn("episode 1", reason)


class SchemaFallbackIntegrationTests(unittest.TestCase):
    """Invalid newer candidates must ride the existing newest→oldest
    fallback; all-invalid populations must exit non-zero."""

    def test_nan_newer_candidate_falls_back_to_older_valid(self):
        root = _run_layout(lambda r: None)
        nested = _nested(root)
        _write_full_run(nested / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
        _write_raw_summary_run(
            nested / f"{LEVEL}-seed-0-timestamp-{TS_NEW}",
            '{"tokens_in": NaN, "success": true}\n\n',
        )
        with _chdir(root), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = _submit.main(_main_config(root))
        self.assertEqual(code, 0)
        diag = err.getvalue()
        self.assertIn(f"[submit-fallback] seed=0", diag)
        self.assertIn(TS_NEW, diag)
        self.assertIn("not a finite number", diag)
        leaderboard = json.loads(
            (root / "tmp" / f"{LEVEL}-leaderboard.json").read_text(encoding="utf-8")
        )
        self.assertEqual(leaderboard["leaderboard_results"]["num_seeds"], 1)
        self.assertEqual(leaderboard["leaderboard_results"]["timestamp"], TS_OLD)

    def test_overflow_newer_candidate_falls_back_to_legacy_valid(self):
        root = _run_layout(lambda r: None)
        nested = _nested(root)
        _write_full_run(nested / f"{LEVEL}-seed-0")
        _write_raw_summary_run(
            nested / f"{LEVEL}-seed-0-timestamp-{TS_NEW}",
            '{"per_cube_success": [1e999]}\n\n',
        )
        runs, unusable = _submit.collect_task_runs(nested, LEVEL)
        self.assertEqual(unusable, [])
        self.assertEqual(Path(runs[0]["path"]).name, f"{LEVEL}-seed-0")
        self.assertIn("not a finite number", runs[0]["skipped"][0]["reason"])

    def test_all_candidates_invalid_main_returns_2(self):
        root = _run_layout(lambda r: None)
        nested = _nested(root)
        _write_raw_summary_run(
            nested / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", "null\n\n", ts=TS_OLD
        )
        _write_raw_summary_run(
            nested / f"{LEVEL}-seed-0-timestamp-{TS_NEW}",
            '{"tokens_in": Infinity}\n\n',
        )
        with _chdir(root), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            code = _submit.main(_main_config(root))
        self.assertEqual(code, 2)
        self.assertIn("no usable candidate", err.getvalue())
        self.assertFalse((root / "tmp" / f"{LEVEL}-leaderboard.json").exists())

    def test_valid_historical_full_schema_still_passes(self):
        # The 13-field record run.py actually writes must stay usable.
        root = _run_layout(lambda r: None)
        nested = _nested(root)
        _write_full_run(nested / f"{LEVEL}-seed-0-timestamp-{TS_OLD}", ts=TS_OLD)
        runs, unusable = _submit.collect_task_runs(nested, LEVEL)
        self.assertEqual(unusable, [])
        _submit.verify_consistency(runs)
        metrics = _submit.compute_task_metrics(runs)
        self.assertEqual(metrics["num_seeds"], 1)
        self.assertEqual(metrics["mean_top1_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
