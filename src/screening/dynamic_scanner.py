"""Dynamic Universe Scanner — real-time top-gainers selection for the
live engine.

Replaces the static 10-symbol watchlist with full-market coverage via REST
snapshots. Every scan tick (default 60s), the scanner:

  1. Pulls Alpaca multi-symbol snapshots for the entire micro-cap universe
     (3,500+ symbols from `src.backtest.extended_universe.ALL_MICRO_CAP_SYMBOLS`).
  2. Applies the parabolic-screening criteria from config/settings.yaml:
        - intraday gain in [min_percent_gain, max_percent_gain]
        - price in [min_price, max_price]
        - volume >= min_volume
  3. Ranks survivors by a setup-quality score (higher = more parabolic-like).
  4. Returns the top-N symbols.

The engine then diffs the result against its current WS subscription set,
unsubscribes departed symbols, and subscribes new ones — staying under the
Alpaca IEX free-tier cap (~30 symbols across all channels = 10 with
trades+quotes+bars).

Designed to be cheap: a full 3,500-symbol snapshot completes in ~3.5s on
broadband, so the scanner runs comfortably inside a 60s tick.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.data.alpaca_client import AlpacaClient
from src.utils.config import CONFIG
from src.utils.logger import logger


def _load_universe() -> List[str]:
    """Load ALL_MICRO_CAP_SYMBOLS without importing the backtest package
    (which pulls in matplotlib via visualizer.py).

    Direct-loads ``src/backtest/extended_universe.py`` as a standalone module.
    """
    path = Path("src/backtest/extended_universe.py").resolve()
    spec = importlib.util.spec_from_file_location("_extended_universe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    syms = getattr(mod, "ALL_MICRO_CAP_SYMBOLS", [])
    return [s.upper() for s in syms]


class DynamicScanner:
    """Periodically scans the full micro-cap universe and selects the top
    parabolic-candidate symbols for live streaming."""

    def __init__(self, alpaca: AlpacaClient, universe: Optional[List[str]] = None):
        self.alpaca = alpaca
        self.universe: List[str] = universe if universe is not None else _load_universe()
        logger.info(f"DynamicScanner: universe = {len(self.universe)} symbols")

    def _score(
        self,
        percent_gain: float,
        percent_from_high: float,
        volume: float,
        price: float,
    ) -> float:
        """Setup-quality score in roughly [0, 100].

        Higher when (a) gain is in the sweet-spot range, (b) price is close
        to day high (parabolic top, primed for reversal), (c) volume is
        elevated. Mirrors the existing ParabolicScreener heuristic; kept
        simple and bounded so the scanner is deterministic and cheap.
        """
        # Gain quality: sweet spot 80-200%, drops above 300% (too volatile),
        # drops below 60% (not parabolic enough).
        if percent_gain < 60:
            gain_score = 0.0
        elif percent_gain <= 200:
            gain_score = 1.0 - abs(percent_gain - 120) / 120  # peak at 120%
            gain_score = max(0.0, min(1.0, gain_score))
        else:
            gain_score = max(0.0, 1.0 - (percent_gain - 200) / 300)

        # Proximity-to-high quality: smaller distance is better (within 5%).
        proximity_score = max(0.0, 1.0 - percent_from_high / 5.0)

        # Volume quality: log-scale above min_volume.
        import math
        min_vol = float(CONFIG.screening.min_volume)
        if volume < min_vol:
            vol_score = 0.0
        else:
            # +1 score per decade above min_volume, cap at 1.0
            vol_score = min(1.0, math.log10(max(volume / min_vol, 1.0)) / 2.0)

        # Weighted sum * 100 for human readability.
        return 100.0 * (0.5 * gain_score + 0.3 * proximity_score + 0.2 * vol_score)

    def scan(self, max_candidates: int = 10) -> List[Tuple[str, float, Dict]]:
        """Run a single scan over the universe.

        Returns a list of (symbol, score, snapshot_metrics) tuples sorted by
        score descending. Length <= max_candidates.
        """
        sc = CONFIG.screening
        snaps = self.alpaca.get_snapshots(self.universe)
        results: List[Tuple[str, float, Dict]] = []
        for sym, snap in snaps.items():
            db = snap.get("daily_bar") if snap else None
            if not db:
                continue
            open_p = db.get("open") or 0.0
            high_p = db.get("high") or 0.0
            close_p = db.get("close") or 0.0
            volume = db.get("volume") or 0.0
            # Use latest trade if available, else daily close, for current price.
            current = snap.get("latest_trade_price") or close_p
            if open_p <= 0 or current <= 0 or high_p <= 0:
                continue

            percent_gain = (current - open_p) / open_p * 100.0
            percent_from_high = (high_p - current) / high_p * 100.0 if high_p > 0 else 100.0

            # Apply config filters (same as ParabolicScreener.is_valid_setup).
            if percent_gain < sc.min_percent_gain or percent_gain > sc.max_percent_gain:
                continue
            if current < sc.min_price or current > sc.max_price:
                continue
            if volume < sc.min_volume:
                continue

            score = self._score(percent_gain, percent_from_high, volume, current)
            results.append((sym, score, {
                "percent_gain": percent_gain,
                "percent_from_high": percent_from_high,
                "volume": volume,
                "current_price": current,
            }))

        results.sort(key=lambda x: -x[1])
        top = results[:max_candidates]
        logger.info(
            f"DynamicScanner: scanned {len(snaps)} snapshots, "
            f"{len(results)} passed filters, top={[(s, round(sc_, 1)) for s, sc_, _ in top]}"
        )
        return top

    def select_symbols(self, max_candidates: int = 10) -> Set[str]:
        """Convenience: just the symbol set, ready to diff against current
        WS subscriptions."""
        return {s for s, _, _ in self.scan(max_candidates=max_candidates)}
