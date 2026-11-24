"""
Configuration loading (Master Prompt Section 4/7/16): reads configs/*.yaml
and turns them into the keyword arguments the rest of the codebase already
accepts, rather than introducing a second, parallel configuration system.

This module is intentionally thin: AISOSAgent, the Scheduler, the Interrupt
Controller, and every experiment's main() already expose every tunable
value as a constructor/function argument (this was true before this module
existed -- see aisos/core/adapter.py's __init__ signature). What was missing
was a way to change those values from a file instead of editing source; this
module is exactly that, nothing more.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Packaged fallback (SDK Phase C, Decision 9). CONFIGS_DIR resolves to a
# sibling of the installed `aisos` package, which exists in a source
# checkout / editable install but NOT in a built wheel installed into
# site-packages -- previously making from_config() and backend_from_profile()
# raise FileNotFoundError there. This fallback is strictly additive:
# CONFIGS_DIR is still consulted first, so every existing research script
# and experiment resolves exactly the same file it always did.
PACKAGED_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def load_yaml(name_or_path: str) -> dict:
    """`name_or_path` may be a bare filename resolved against configs/
    (e.g. "default.yaml") or an absolute/relative path to any YAML file.

    Resolution order for a bare filename: an existing relative path as
    given, then the repo-root configs/ directory (authoritative for
    research), then the packaged aisos/configs/ copy (wheel installs).
    """
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        candidate = CONFIGS_DIR / name_or_path
        path = candidate if candidate.exists() else PACKAGED_CONFIGS_DIR / name_or_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def agent_kwargs_from_config(
    default_path: str = "default.yaml",
    models_path: str = "models.yaml",
    overrides: Optional[dict] = None,
) -> dict[str, Any]:
    """Flattens configs/default.yaml + configs/models.yaml into the exact
    keyword arguments AISOSAgent.__init__ accepts. `overrides` is applied
    last (e.g. a test or experiment that wants everything from the file
    except tau_light).
    """
    d = load_yaml(default_path)
    m = load_yaml(models_path)

    sched = d.get("scheduler", {})
    interrupt = d.get("interrupt", {})
    memory = d.get("memory", {})
    promo = d.get("promotion", {})
    fleet = d.get("fleet", {})
    dream = m.get("dream", {})

    from aisos.core.scheduler import SleepPressureWeights
    from aisos.dream.dream_phase import DreamBudget

    weights_dict = sched.get("weights")
    weights = SleepPressureWeights(**weights_dict) if weights_dict else None

    kwargs: dict[str, Any] = dict(
        weights=weights,
        tau_light=sched.get("tau_light", 0.35),
        tau_deep=sched.get("tau_deep", 0.65),
        atomic_unit_duration_ms=interrupt.get("atomic_unit_duration_ms", 50.0),
        max_defer_ms=interrupt.get("max_defer_ms", 100.0),
        dream_budget=DreamBudget(
            n_seeds=dream.get("n_seeds", 10),
            clip_bound=dream.get("clip_bound", 0.05),
            lr=dream.get("lr", 0.02),
        ),
        theta_dims=m.get("theta_dims", 8),
        segment_size=memory.get("segment_size", 8),
        max_versions_per_subject=memory.get("max_versions_per_subject", 3),
        max_allowed_drift=promo.get("max_allowed_drift", 0.6),
        drift_baseline_min=promo.get("drift_baseline_min", 20),
        max_promotions_per_cycle=promo.get("max_promotions_per_cycle", 25),
        confidence_threshold=promo.get("confidence_threshold", 0.55),
        max_promotion_rounds=promo.get("max_promotion_rounds", 4),
        consistency_tolerance=promo.get("consistency_tolerance", 0),
        replica_id=fleet.get("replica_id", 0),
    )
    if overrides:
        kwargs.update(overrides)
    return kwargs


def load_experiments_config(path: str = "experiments.yaml") -> dict:
    return load_yaml(path)
