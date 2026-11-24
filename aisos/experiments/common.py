"""Shared helpers for the experiment scripts (v2 Section 7)."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aisos.metrics import mean_ci95

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def summarize_by_condition(rows: list[dict], metric_keys: list[str]) -> dict:
    """Group a flat list of per-seed metric dicts by 'condition' and report
    mean + 95% CI half-width per metric (v2 Section 7.6 reporting standard).
    """
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    out = {}
    for cond, rs in by_cond.items():
        entry = {"n_seeds": len(rs)}
        for key in metric_keys:
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                m, half = mean_ci95(vals)
                entry[key] = {"mean": m, "ci95_half_width": half, "n": len(vals)}
            else:
                entry[key] = None
        out[cond] = entry
    return out


def save_json(name: str, payload: dict) -> Path:
    ensure_results_dir()
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str))
    return path


def print_table(title: str, summary: dict, metric_keys: list[str]) -> None:
    print(f"\n=== {title} ===")
    header = "condition".ljust(24) + "".join(k[:16].ljust(18) for k in metric_keys)
    print(header)
    print("-" * len(header))
    for cond, entry in summary.items():
        row = cond.ljust(24)
        for k in metric_keys:
            v = entry.get(k)
            if v is None:
                row += "n/a".ljust(18)
            else:
                row += f"{v['mean']:.4f}±{v['ci95_half_width']:.4f}".ljust(18)
        print(row)
