"""Unit tests for StreamingBuffer VWAP session reset.

The plan suggested ``from src.utils.types import TickData`` — that module does
not exist in this codebase. ``TickData`` is defined in ``src.data.polars_engine``
itself, but the reset_session test does not need it: we manipulate buffer state
directly, never adding ticks.
"""
from datetime import datetime, timedelta

from src.data.polars_engine import StreamingBuffer, PolarsSignalEngine


def test_reset_session_zeros_vwap_state():
    """reset_session() should zero all VWAP cumulative state."""
    b = StreamingBuffer("AMC")
    b.cum_pv = 5000.0
    b.cum_vol = 500.0
    b.current_vwap = 10.0
    b.last_bar_time = datetime(2026, 5, 16, 15, 59, 0)

    b.reset_session()

    assert b.cum_pv == 0.0
    assert b.cum_vol == 0.0
    assert b.current_vwap == 0.0
    assert b.last_bar_time is None


def test_cleanup_old_data_resets_buffer_at_session_boundary():
    """When last_bar_time is on a previous calendar day, cleanup_old_data should call reset_session."""
    engine = PolarsSignalEngine()
    engine.register_symbol("AMC")
    buf = engine.buffers["AMC"]
    buf.cum_pv = 5000.0
    buf.cum_vol = 500.0
    buf.current_vwap = 10.0
    # Mark the last bar as belonging to yesterday so cleanup sees a session boundary.
    buf.last_bar_time = datetime.now() - timedelta(days=1)

    engine.cleanup_old_data()

    assert buf.cum_pv == 0.0
    assert buf.cum_vol == 0.0
    assert buf.current_vwap == 0.0


def test_cleanup_old_data_preserves_state_within_session():
    """When the buffer is still in the same calendar day, VWAP state must NOT reset."""
    engine = PolarsSignalEngine()
    engine.register_symbol("AMC")
    buf = engine.buffers["AMC"]
    buf.cum_pv = 5000.0
    buf.cum_vol = 500.0
    buf.current_vwap = 10.0
    buf.last_bar_time = datetime.now()

    engine.cleanup_old_data()

    assert buf.cum_pv == 5000.0
    assert buf.cum_vol == 500.0
    assert buf.current_vwap == 10.0
