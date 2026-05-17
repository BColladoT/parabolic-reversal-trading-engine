"""Tests for the entry-feature snapshot attached to TradeSignal."""
from datetime import datetime


def test_entry_signal_carries_required_features():
    from src.execution.signal_engine import TradeSignal, SignalType

    # Verify the dataclass has a `features` field that defaults to empty dict
    sig = TradeSignal(
        symbol="X",
        signal_type=SignalType.ENTRY_SHORT,
        timestamp=datetime.now(),
        price=10.0,
        confidence=0.8,
        vwap=9.0,
        atr=0.2,
        volume_ratio=3.0,
        volume_exhaustion=True,
    )
    assert isinstance(sig.features, dict)
    assert sig.features == {}


def test_entry_signal_features_contains_expected_keys_when_set():
    from src.execution.signal_engine import TradeSignal, SignalType

    sig = TradeSignal(
        symbol="X",
        signal_type=SignalType.ENTRY_SHORT,
        timestamp=datetime.now(),
        price=10.0,
        confidence=0.8,
        vwap=9.0,
        atr=0.2,
        volume_ratio=3.0,
        volume_exhaustion=True,
        features={
            "vwap_extension": 0.111,
            "volume_ratio": 3.0,
            "atr_pct": 0.02,
            "time_of_day_min": 60.0,
            "day_of_week": 0.0,
            "factors_count": 3.0,
        },
    )
    for k in [
        "vwap_extension",
        "volume_ratio",
        "atr_pct",
        "time_of_day_min",
        "day_of_week",
        "factors_count",
    ]:
        assert k in sig.features


def test_entry_signal_features_populated_by_engine(monkeypatch):
    """When the engine emits an ENTRY_SHORT, the signal must carry a populated features dict.

    Stubs out internal indicator checks so we can drive the generator without a real
    data stream. The exact numeric values are checked relative to the inputs.
    """
    from src.execution import signal_engine as se
    from src.execution.signal_engine import (
        ParabolicSignalEngine,
        SignalType,
        VolumeProfile,
    )
    from src.screening.screener import ScreenedAsset

    # Build a minimal stub for the data engine
    class _StubDataEngine:
        buffers: dict = {}

        def get_signal_data(self, symbol: str):
            return {
                "last_price": 12.0,
                "vwap": 10.0,
                "atr": 0.24,
            }

    engine = ParabolicSignalEngine(_StubDataEngine())

    # Patch helper checks so we don't need real history
    monkeypatch.setattr(engine, "_check_momentum_divergence", lambda symbol: True)
    monkeypatch.setattr(engine, "_detect_absorption", lambda symbol, price: True)

    # Construct screened asset stub. ScreenedAsset may have many fields; we set only what
    # generate_entry_signal reads (symbol, current_price, day_high).
    class _AssetStub:
        symbol = "TEST"
        current_price = 12.0
        day_high = 12.1

    asset = _AssetStub()

    # Volume profile with strong exhaustion (low ratio)
    profile = VolumeProfile()
    profile.peak_volume = 100_000
    profile.current_volume_5min = 30_000
    profile.volume_ratio = 0.30  # exhausted, below typical entry threshold

    # Prime day_high so price_proximity check passes
    engine.day_highs["TEST"] = 12.1

    signal = engine.generate_entry_signal(asset, profile)

    # The engine may decline to emit if thresholds aren't met; for this test our config
    # values should result in an emission. If it doesn't emit, the test below will fail
    # loudly with an informative message rather than silently passing.
    assert signal is not None, "engine did not emit an ENTRY_SHORT signal under exhaustion stub"
    assert signal.signal_type == SignalType.ENTRY_SHORT

    feats = signal.features
    assert isinstance(feats, dict) and feats, "features dict must be populated at emission"
    for k in [
        "vwap_extension",
        "volume_ratio",
        "atr_pct",
        "time_of_day_min",
        "day_of_week",
        "factors_count",
    ]:
        assert k in feats, f"missing key {k!r} in features snapshot"

    # vwap_extension is (price - vwap) / vwap = (12 - 10)/10 = 0.20
    assert abs(feats["vwap_extension"] - 0.20) < 1e-9
    # volume_ratio is what we passed via the profile
    assert abs(feats["volume_ratio"] - 0.30) < 1e-9
    # atr_pct = atr / price = 0.24 / 12 = 0.02
    assert abs(feats["atr_pct"] - 0.02) < 1e-9
    # factors_count is a positive int-as-float reflecting how many exhaustion factors fired
    assert feats["factors_count"] >= 1.0
