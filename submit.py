"""Generates the final JSON object linking results to environment metadata."""

import json
import math
import sys
import numpy as np
from pathlib import Path
from typing import Any

from dataclasses import dataclass
from simple_parsing import ArgumentParser
from builderbench.create_task_data import *

from output_paths import iter_candidates_by_seed, parse_run_dirname

@dataclass
class Args:
    results_dir: str
    level_id: str
    model_id: str = "naive"
    agent_name: str = "Qwen3-235B-A22B-Instruct-2507"
    website_url: str = "N/A"

def verify_consistency(runs: list[dict[str, Any]]) -> None:
    """
    Sanity Check: Ensures all seeds ran on the exact same task version,
    code commit, and model configuration.
    """
    if not runs:
        return

    first_run = runs[0]
    first_meta = first_run.get("run_metadata")
    if not first_meta:
        raise ValueError(f"Missing metadata for Run: {first_run}")

    ref_sha = first_meta.get("task_version", {}).get("task_file_sha256")
    ref_commit = first_meta.get("builderbench_git_commit")
    ref_model = first_meta.get("model", {}).get("model_id")

    for run in runs[1:]:
        meta = run.get("run_metadata")
        if not meta:
            raise ValueError(f"Missing metadata for Run: {run}")

        # Check Task Version
        curr_sha = meta.get("task_version", {}).get("task_file_sha256")
        if curr_sha != ref_sha:
            raise ValueError(
                f"Consistency Error! Seed {run['seed']} used task SHA {curr_sha}, "
                f"but seed {first_run['seed']} used {ref_sha}. "
                "Cannot aggregate results from different task versions."
            )

        # Check Git Commit
        curr_commit = meta.get("builderbench_git_commit")
        if curr_commit != ref_commit:
            raise ValueError(
                f"Consistency Error! Seed {run['seed']} used commit {curr_commit}, "
                f"but seed {first_run['seed']} used {ref_commit}."
             )
        # Check Model ID
        curr_model = meta.get("model", {}).get("model_id")
        if curr_model != ref_model:
            raise ValueError(
                f"Consistency Error! Seed {run['seed']} used model {curr_model}, "
                f"but seed {first_run['seed']} used {ref_model}."
            )

def load_eval_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    chunks = [chunk for chunk in content.split("\n\n") if chunk.strip()]
    return [json.loads(chunk) for chunk in chunks]


def _episode_problem(episode: Any, index: int) -> str | None:
    """Return a rejection reason if ``episode`` cannot be consumed safely
    by :func:`compute_task_metrics`, else ``None``.

    Validation mirrors only what the metrics code actually reads:

    * the episode must be a JSON object (``ep.get`` is called on it);
    * ``tokens_in`` / ``tokens_out`` go through ``int(value or 0)``, so
      anything that conversion rejects (non-numeric strings, NaN,
      infinity) is rejected here instead of crashing aggregation;
    * ``per_cube_success`` is summed, so it must be a list of finite
      numeric/boolean scalars;
    * ``per_cube_mask`` is iterated and zipped, so it must be a list or
      absent/null.

    Fields the metrics code never reads (``episode_idx``, ``steps``,
    ``return``, ...) are not required, and every consumed field keeps its
    existing ``.get`` default — an episode missing them all is still a
    valid all-zero record, matching historical behaviour.
    """
    if not isinstance(episode, dict):
        return f"episode {index} is not a JSON object"
    for key in ("tokens_in", "tokens_out"):
        value = episode.get(key, 0) or 0
        if isinstance(value, float) and not math.isfinite(value):
            return f"episode {index} {key} is not a finite number"
        try:
            int(value)
        except (TypeError, ValueError, OverflowError):
            return f"episode {index} {key} cannot be converted to an integer token count"
    per_cube_success = episode.get("per_cube_success", [])
    if not isinstance(per_cube_success, list):
        return f"episode {index} per_cube_success is not a list"
    for position, value in enumerate(per_cube_success):
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            return (
                f"episode {index} per_cube_success[{position}] is not numeric"
            )
        if isinstance(value, float) and not math.isfinite(value):
            return (
                f"episode {index} per_cube_success[{position}] is not a finite number"
            )
    per_cube_mask = episode.get("per_cube_mask")
    if per_cube_mask is not None and not isinstance(per_cube_mask, list):
        return f"episode {index} per_cube_mask is not a list"
    return None


def _load_run_record(run_dir: Path, task_name: str):
    """Probe a candidate directory for the minimum records this pipeline
    requires.  Returns ``(usable: bool, reason: str, record: dict | None)``.

    ``usable`` claims only that the candidate carries the fields needed to
    run :func:`verify_consistency` and :func:`compute_task_metrics`:

    * ``run_metadata.json`` — exists, parses as JSON to a truthy dict
      (matching the ``if not meta: raise`` contract that verify_consistency
      already enforces across seeds).
    * ``eval_summary.jsonl`` — exists, non-empty, parses as
      ``"\\n\\n"``-separated JSON objects, and yields at least one episode.
      Every episode must also pass :func:`_episode_problem`, which checks
      exactly the fields ``compute_task_metrics`` consumes (object shape,
      integer-convertible finite token counts, finite numeric
      ``per_cube_success`` entries, list-or-absent ``per_cube_mask``).
      Non-finite values (NaN, ±Infinity, JSON numbers that overflow to
      infinity) are rejected outright — never coerced to zero.
    * ``run_config.json`` — **optional** if absent (historical runs may not
      have written it — the pre-fix code also tolerated absence), but if
      present must parse as JSON.  This keeps compatibility with the prior
      behaviour.

    ``usable`` is *not* a claim that the underlying experiment finished.
    The current ``eval_summary.jsonl`` schema carries no end-of-run
    sentinel and ``run_metadata.json`` does not record the planned
    ``num_episodes``, so a run that crashed after writing episode k looks
    identical to a completed k-episode run.  Detecting this requires a
    writer-side schema change (out of scope for this fallback fix).
    """
    parsed = parse_run_dirname(run_dir.name)
    if parsed is None:
        return False, "unrecognised directory name", None
    _parsed_task, seed, timestamp = parsed

    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return False, "run_metadata.json missing", None
    try:
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"run_metadata.json not parseable ({exc.msg})", None
    except OSError as exc:
        return False, f"run_metadata.json unreadable ({exc.strerror})", None
    if not isinstance(run_metadata, dict) or not run_metadata:
        return False, "run_metadata.json empty or wrong type", None

    eval_path = run_dir / "eval_summary.jsonl"
    if not eval_path.exists():
        return False, "eval_summary.jsonl missing", None
    try:
        content = eval_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return False, f"eval_summary.jsonl unreadable ({exc.strerror})", None
    if not content:
        return False, "eval_summary.jsonl empty", None
    chunks = [c for c in content.split("\n\n") if c.strip()]
    if not chunks:
        return False, "eval_summary.jsonl contains no records", None
    try:
        episodes = [json.loads(chunk) for chunk in chunks]
    except json.JSONDecodeError as exc:
        return False, f"eval_summary.jsonl not parseable ({exc.msg})", None
    for index, episode in enumerate(episodes):
        problem = _episode_problem(episode, index)
        if problem is not None:
            return False, f"eval_summary.jsonl invalid: {problem}", None

    run_config = None
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, f"run_config.json not parseable ({exc.msg})", None
        except OSError as exc:
            return False, f"run_config.json unreadable ({exc.strerror})", None

    record = {
        "task": task_name,
        "seed": seed,
        "path": str(run_dir),
        "timestamp": timestamp,
        "eval_summary": episodes,
        "run_config": run_config,
        "run_metadata": run_metadata,
    }
    return True, "", record


def collect_task_runs(results_dir: Path, task_name: str):
    """Return ``(runs, unusable_seeds)``.

    ``runs`` is one record per seed.  For each seed we walk candidates
    newest→oldest (see :func:`output_paths.iter_candidates_by_seed`) and
    keep the first candidate that passes :func:`_load_run_record`.  Any
    candidate skipped on the way to the winner is recorded in
    ``record["skipped"]`` so callers can surface the fallback.

    ``unusable_seeds`` is one entry per seed whose every candidate was
    rejected; each entry lists the paths and reasons.  ``main()`` must
    treat a non-empty list as a submission error rather than silently
    aggregating fewer seeds.
    """
    candidates = (d for d in results_dir.glob(f"{task_name}-seed-*/") if d.is_dir())
    grouped = iter_candidates_by_seed(candidates)
    runs: list[dict[str, Any]] = []
    unusable: list[dict[str, Any]] = []
    for seed, group in grouped.items():
        skipped: list[dict[str, str]] = []
        chosen: dict[str, Any] | None = None
        for run_dir in group:
            ok, reason, record = _load_run_record(run_dir, task_name)
            if ok:
                chosen = record
                break
            skipped.append({"path": str(run_dir), "reason": reason})
        if chosen is None:
            unusable.append({"seed": seed, "candidates": skipped})
            continue
        chosen["skipped"] = skipped
        runs.append(chosen)
    return runs, unusable


def compute_task_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def compute_progress(per_cube_success, per_cube_mask):
        if per_cube_mask:
            filtered = [s for s, m in zip(per_cube_success, per_cube_mask) if m]
            total = sum(1 for m in per_cube_mask if m)
        else:
            filtered = per_cube_success
            total = len(per_cube_success)
        return (sum(filtered) / total) if total else 0.0

    per_seed = []
    for run in runs:
        episodes = run["eval_summary"]
        if not episodes:
            continue
                
        top1_success = any(bool(ep.get("success", False)) for ep in episodes)

        max_progress_in_run = 0.0
        num_input_tokens = 0
        num_output_tokens = 0
    
        for episode in episodes:
            num_input_tokens += int(episode.get("tokens_in", 0) or 0)
            num_output_tokens += int(episode.get("tokens_out", 0) or 0)

            per_cube_success = episode.get("per_cube_success", [])
            per_cube_mask = episode.get("per_cube_mask")
            progress = compute_progress(per_cube_success, per_cube_mask)
            max_progress_in_run = max(max_progress_in_run, progress)

        final_ep = episodes[-1]
        final_success = bool(final_ep.get("success", False))
        final_pcs = final_ep.get("per_cube_success", [])
        final_masks = final_ep.get("per_cube_mask")
        final_progress = compute_progress(final_pcs, final_masks)

        per_seed.append(
            {
                "seed": run.get("seed"),
                "top1_success": top1_success,
                "top1_progress": max_progress_in_run,
                "final_success": final_success,
                "final_progress": final_progress,
                "input_tokens": num_input_tokens,
                "output_tokens": num_output_tokens,
                "num_episodes": len(episodes),
            }
        )

    num_seeds = len(per_seed)
    if num_seeds == 0:
        return {"num_seeds": 0}

    # Aggregate
    mean_final_success = np.mean( [x["final_success"] for x in per_seed] )
    mean_top1_success = np.mean( [x["top1_success"] for x in per_seed] )
    mean_final_progress = np.mean( [x["final_progress"] for x in per_seed] )
    mean_top1_progress = np.mean( [x["top1_progress"] for x in per_seed] )
    mean_input_tokens_per_run = np.mean( [x["input_tokens"] for x in per_seed] )
    mean_output_tokens_per_run = np.mean( [x["output_tokens"] for x in per_seed] )
    mean_num_episodes = np.mean( [x["num_episodes"] for x in per_seed] )

    std_final_success = np.std( [x["final_success"] for x in per_seed] )
    std_top1_success = np.std( [x["top1_success"] for x in per_seed] )
    std_final_progress = np.std( [x["final_progress"] for x in per_seed] )
    std_top1_progress = np.std( [x["top1_progress"] for x in per_seed] )
    std_input_tokens_per_run = np.std( [x["input_tokens"] for x in per_seed] )
    std_output_tokens_per_run = np.std( [x["output_tokens"] for x in per_seed] )
    std_num_episodes = np.std( [x["num_episodes"] for x in per_seed] )

    return {
        "num_seeds": num_seeds,
        "mean_final_success_rate": mean_final_success,
        "mean_top1_success_rate": mean_top1_success,
        "mean_final_progress": mean_final_progress,
        "mean_top1_progress": mean_top1_progress,
        "mean_num_episodes": mean_num_episodes,
        "mean_input_tokens_per_run": mean_input_tokens_per_run,
        "mean_output_tokens_per_run": mean_output_tokens_per_run,
        "std_final_success_rate": std_final_success,
        "std_top1_success_rate": std_top1_success,
        "std_final_progress": std_final_progress,
        "std_top1_progress": std_top1_progress,
        "std_num_episodes": std_num_episodes,
        "std_input_tokens_per_run": std_input_tokens_per_run,
        "std_output_tokens_per_run": std_output_tokens_per_run,
    }

def main(config: Args) -> int:
    results_dir = ( Path(config.results_dir) / config.agent_name / config.model_id ).expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    runs, unusable = collect_task_runs(results_dir, config.level_id)
    if not runs and not unusable:
        raise SystemExit(f"No runs found for task '{config.level_id}' in {results_dir}")

    # Visible diagnostics for every candidate skipped during the newest→
    # oldest fallback within a seed, so an incomplete newer run cannot
    # silently mask an older complete one.
    for run in runs:
        for skipped in run.get("skipped", []):
            print(
                f"[submit-fallback] seed={run['seed']} skipped {skipped['path']} "
                f"— reason: {skipped['reason']}",
                file=sys.stderr,
            )
    for entry in unusable:
        listing = "; ".join(
            f"{item['path']} ({item['reason']})" for item in entry["candidates"]
        )
        print(
            f"[submit-fallback] seed={entry['seed']} has no usable candidate: {listing}",
            file=sys.stderr,
        )
    if unusable:
        print(
            f"[submit-fallback] refusing to submit: {len(unusable)} seed(s) "
            "lack a usable candidate; aggregate would be short.",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(runs)} run(s) for {config.level_id} in {results_dir}")

    try:
        verify_consistency(runs)
        print("Consistency check passed (Task SHA, Commit, Model ID match).")
    except ValueError as e:
        print(f"Consistency Check Failed: {e}")
        return 1

    metrics = compute_task_metrics(runs)
    if metrics["num_seeds"] == 0:
        print("No valid episodes found in runs.")
        return 0

    meta = runs[0].get("run_metadata", {})
    
    # Safely extract nested keys
    task_ver = meta.get("task_version", {})
    deps = meta.get("dependency_versions", {})
    model_info = meta.get("model", {})
    
    leaderboard = {
        "leaderboard_results": {
            "level_id": task_ver.get("level_id", "unknown"),
            "agent_name": config.agent_name,
            "model_id": config.model_id,
            "website_url": config.website_url,
            "timestamp": meta.get("run_timestamp"),
            **metrics
        },
        "reproducibility": {
            "builderbench_git_commit": meta.get("builderbench_git_commit"),
            "task_version": {
                **task_ver,
            },
            "dependencies": {
                **deps
            },
        },
    }
    
    Path("tmp").mkdir(exist_ok=True)
    output_path = Path("tmp") / f"{config.level_id}-leaderboard.json"
    output_path.write_text(json.dumps(leaderboard, indent=4), encoding="utf-8")
    print(f"\nLeaderboard results saved to: {output_path}")

    return 0

if __name__ == "__main__":
    parser = ArgumentParser(description="Collect results for a task.")
    parser.add_arguments(Args, dest="config")
    args = parser.parse_args()

    raise SystemExit(main(args.config))