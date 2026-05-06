"""
Metrics writer for the real-time training dashboard.

Appends one JSONL line per training iteration. The dashboard server
reads this file and streams charts to the browser.

This module is intentionally lightweight — no external dependencies,
no network calls. A write failure must never crash training.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _json_safe(obj):
    """Convert numpy types and handle NaN/inf for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_metrics_line(
    metrics_path: Path,
    iteration: int,
    result: Dict[str, Any],
    fold: int,
    start_time: datetime,
    target_timesteps: int,
) -> None:
    """
    Append one JSONL line with training metrics for the dashboard.

    Args:
        metrics_path: Path to the .jsonl file
        iteration: Current training iteration index
        result: The dict returned by algo.train()
        fold: Current WFO fold number
        start_time: datetime.now() captured before the training loop
        target_timesteps: Total timesteps target for this fold
    """
    try:
        # Extract from RLlib result (try top-level, then nested sampler_results)
        sampler = result.get("sampler_results", {})

        line = {
            "iteration": iteration,
            "fold": fold,
            "timesteps": result.get("timesteps_total", 0),
            "target_timesteps": target_timesteps,
            "episode_reward_mean": _get_nested(result, sampler, "episode_reward_mean"),
            "episode_reward_max": _get_nested(result, sampler, "episode_reward_max"),
            "episode_reward_min": _get_nested(result, sampler, "episode_reward_min"),
            "episodes_this_iter": result.get("episodes_this_iter", 0),
            "episode_len_mean": _get_nested(result, sampler, "episode_len_mean"),
            # Custom callback metrics (injected by WarmupCallback.on_train_result)
            "phase": result.get("phase", "unknown"),
            "phase_num": result.get("phase_num", 0),
            "actor_lr": result.get("actor_lr"),
            "scheduled_alpha": result.get("scheduled_alpha"),
            # SAC learner diagnostics (losses, Q-values, entropy)
            **_extract_learner_stats(result),
            # Timing
            "elapsed_seconds": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
        }

        # Sanitize any remaining NaN/inf values before writing
        line = _sanitize_dict(line)

        with open(metrics_path, "a") as f:
            f.write(json.dumps(line, default=_json_safe) + "\n")

    except Exception as e:
        # Never crash training because of a dashboard write failure.
        # But DO log the error so we can diagnose dashboard issues.
        import logging
        logging.getLogger("dashboard.metrics").debug(
            f"Metrics write failed (iter={iteration}): {e}"
        )


def write_trade_log(
    trades_path: Path,
    trades: list,
    fold: int,
    episode_symbol: str = "",
    episode_date: str = "",
) -> None:
    """
    Append completed trades from an episode to the trades JSONL file.

    Args:
        trades_path: Path to the trades .jsonl file
        trades: List of TradeRecord objects (or dicts with same fields)
        fold: Current WFO fold number
        episode_symbol: Fallback symbol if trade doesn't have one
        episode_date: Episode date string
    """
    try:
        with open(trades_path, "a") as f:
            for t in trades:
                # Support both TradeRecord objects and dicts
                get = (lambda k, d=None: getattr(t, k, d)) if hasattr(t, 'pnl') else (lambda k, d=None: t.get(k, d))

                entry_time = get('entry_time')
                exit_time = get('timestamp')

                line = {
                    "fold": fold,
                    "symbol": get('symbol') or episode_symbol,
                    "date": episode_date,
                    "entry_price": round(float(get('entry_price', 0)), 2),
                    "exit_price": round(float(get('exit_price', 0)), 2),
                    "entry_time": entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time) if entry_time else None,
                    "exit_time": exit_time.isoformat() if hasattr(exit_time, 'isoformat') else str(exit_time) if exit_time else None,
                    "shares": round(float(get('shares', 0)), 1),
                    "pnl": round(float(get('pnl', 0)), 2),
                    "return_pct": round(float(get('return_pct', 0)), 3),
                    "win": bool(get('win', False)),
                    "bars_held": int(get('bars_held', 0)),
                    "vwap_at_entry": round(float(get('vwap_at_entry', 0)), 2),
                    "mfe": round(float(get('mfe', 0)), 2),
                    "mae": round(float(get('mae', 0)), 2),
                }
                f.write(json.dumps(line, default=_json_safe) + "\n")
    except Exception:
        pass


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN/inf floats with None so JSON is JavaScript-safe."""
    clean = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean


def _extract_learner_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract SAC learner diagnostics from the nested result dict."""
    stats = (result.get("info", {})
             .get("learner", {})
             .get("default_policy", {})
             .get("learner_stats", {}))
    if not stats:
        return {}
    return {
        "actor_loss": stats.get("actor_loss"),
        "critic_loss": stats.get("critic_loss"),
        "alpha_loss": stats.get("alpha_loss"),
        "alpha_value": stats.get("alpha_value"),
        "mean_q": stats.get("mean_q"),
        "max_q": stats.get("max_q"),
        "min_q": stats.get("min_q"),
    }


def _get_nested(
    result: Dict, sampler: Dict, key: str
) -> Optional[float]:
    """Get a metric from top-level result or nested sampler_results."""
    val = result.get(key)
    if val is not None:
        return val
    return sampler.get(key)
