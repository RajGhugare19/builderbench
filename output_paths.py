"""Stdlib-only helpers for the BuilderBench experiment output layout.

Kept as a small module so run.py's path construction and submit.py's
directory discovery can be exercised without importing mujoco, mediapy,
agents, or simple-parsing.

Layout produced (matches README.md#leaderboard-submission):

    outputs/<agent_name>/<model_id>/<level_id>-seed-<seed>-timestamp-<ts>/

where ``<ts>`` is ``YYYY-MM-DD_HH-MM-SS-ffffff``.  ``ffffff`` is the
microsecond component of ``datetime.now()``, which makes same-second
reruns land in different directories on the same machine and lets the
filesystem reject a true collision instead of silently overwriting.

Legacy directories that predate the fix (``<level_id>-seed-<seed>`` with
no timestamp suffix) are still parsed by :func:`parse_run_dirname` and
are treated as the oldest candidate for their seed by
:func:`select_latest_per_seed`.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S-%f"
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}")

RUN_DIR_RE = re.compile(
    r"^(?P<task>.+)-seed-(?P<seed>\d+)"
    r"(?:-timestamp-(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}))?$"
)


def format_timestamp(when: datetime) -> str:
    """Return the wall-clock string used as the timestamp suffix."""
    if not isinstance(when, datetime):
        raise TypeError(f"format_timestamp expects datetime.datetime, got {type(when).__name__}")
    return when.strftime(TIMESTAMP_FMT)


def build_run_dirname(level_id: str, seed: int, timestamp: str) -> str:
    """Compose the leaf directory name for a single experiment run."""
    if not isinstance(level_id, str) or not level_id:
        raise ValueError("level_id must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    if not isinstance(timestamp, str) or not TIMESTAMP_RE.fullmatch(timestamp):
        raise ValueError(
            f"timestamp {timestamp!r} must match YYYY-MM-DD_HH-MM-SS-ffffff"
        )
    return f"{level_id}-seed-{seed}-timestamp-{timestamp}"


def parse_run_dirname(name: str) -> Optional[Tuple[str, int, Optional[str]]]:
    """Return ``(task, seed, timestamp_or_None)`` for a recognised run dir.

    ``timestamp`` is ``None`` for the pre-fix legacy layout.  Names that
    do not fit either shape return ``None`` so callers can skip them.
    """
    match = RUN_DIR_RE.match(name)
    if match is None:
        return None
    return match.group("task"), int(match.group("seed")), match.group("timestamp")


def _rank(path: Path, timestamp: Optional[str]) -> Tuple[int, str, str]:
    """Sort key so a timestamped directory outranks a legacy sibling.

    Ordering rules inside a seed:
      1. Timestamped > legacy.
      2. Later ISO timestamp > earlier ISO timestamp (lexicographic on
         the zero-padded format is equivalent to chronological).
      3. Equal-rank paths fall back to their string form so selection is
         deterministic on filesystems where ``glob`` ordering varies.
    """
    if timestamp is None:
        return (0, "", str(path))
    return (1, timestamp, str(path))


def select_latest_per_seed(paths: Iterable[Path]) -> list[Path]:
    """Collapse an iterable of run directories to one winner per seed.

    Callers should pass already-filtered directory paths; anything whose
    leaf name is unrecognised is silently skipped so unrelated
    sub-directories inside ``outputs/<agent>/<model>/`` do not derail
    discovery.
    """
    winners: dict[int, Tuple[Tuple[int, str, str], Path]] = {}
    for raw in paths:
        candidate = Path(raw)
        parsed = parse_run_dirname(candidate.name)
        if parsed is None:
            continue
        _, seed, timestamp = parsed
        rank = _rank(candidate, timestamp)
        current = winners.get(seed)
        if current is None or rank > current[0]:
            winners[seed] = (rank, candidate)
    return [winners[seed][1] for seed in sorted(winners)]


def iter_candidates_by_seed(paths: Iterable[Path]) -> dict[int, list[Path]]:
    """Group run directories by seed and order each group newest-first.

    Ranking follows :func:`_rank`: timestamped directories rank above a
    legacy (no-timestamp) sibling, and within the timestamped class the
    larger ISO-formatted timestamp string wins.  Ties fall back to the
    path string so ordering is stable across filesystems.

    Unrecognised names are skipped exactly like
    :func:`select_latest_per_seed` does, so ``outputs/<agent>/<model>/``
    may contain scratch sub-directories without disturbing discovery.

    The caller (``submit.py``) is responsible for validating whether each
    candidate carries usable experiment records; this function only
    decides the traversal order.
    """
    grouped: dict[int, list[tuple[tuple[int, str, str], Path]]] = {}
    for raw in paths:
        candidate = Path(raw)
        parsed = parse_run_dirname(candidate.name)
        if parsed is None:
            continue
        _, seed, timestamp = parsed
        grouped.setdefault(seed, []).append((_rank(candidate, timestamp), candidate))
    return {
        seed: [path for _, path in sorted(ranked, key=lambda item: item[0], reverse=True)]
        for seed, ranked in sorted(grouped.items())
    }
