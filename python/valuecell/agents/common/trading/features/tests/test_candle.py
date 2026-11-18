import itertools
import math

import pytest

from valuecell.agents.common.trading.features.candle import SimpleCandleFeatureComputer
from valuecell.agents.common.trading.models import Candle, InstrumentRef


def _candles(closes: list[float]) -> list[Candle]:
    instrument = InstrumentRef(symbol="BTCUSDT", exchange_id="binance")
    return [
        Candle(
            ts=1_700_000_000_000 + i * 60_000,
            instrument=instrument,
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1.0,
            interval="1m",
        )
        for i, close in enumerate(closes)
    ]


def _values(closes: list[float]) -> dict:
    (feature,) = SimpleCandleFeatureComputer().compute_features(_candles(closes))
    return feature.values


def test_rsi_is_100_for_a_straight_run_up():
    # No losing bar in the window used to give RSI = 0 (maximally oversold).
    values = _values([100.0 + i for i in range(40)])
    assert values["rsi"] == pytest.approx(100.0)


def test_rsi_uses_wilder_smoothing():
    closes = [
        44.34,
        44.09,
        44.15,
        43.61,
        44.33,
        44.83,
        45.10,
        45.42,
        45.84,
        46.08,
        45.89,
        46.03,
        45.61,
        46.28,
        46.28,
        46.00,
        46.03,
        46.41,
        46.22,
        45.64,
    ]
    # Reference: Wilder RSI computed by hand (seed = 14-bar simple average,
    # then avg = (prev * 13 + x) / 14).
    deltas = [b - a for a, b in itertools.pairwise(closes)]
    avg_gain = sum(max(d, 0.0) for d in deltas[:14]) / 14
    avg_loss = sum(max(-d, 0.0) for d in deltas[:14]) / 14
    for d in deltas[14:]:
        avg_gain = (avg_gain * 13 + max(d, 0.0)) / 14
        avg_loss = (avg_loss * 13 + max(-d, 0.0)) / 14
    expected = 100 - 100 / (1 + avg_gain / avg_loss)

    assert _values(closes)["rsi"] == pytest.approx(expected, rel=1e-9)


def test_bollinger_bands_use_population_std():
    closes = [100.0 + (i % 5) for i in range(30)]
    window = closes[-20:]
    mean = sum(window) / 20
    std = math.sqrt(sum((c - mean) ** 2 for c in window) / 20)

    values = _values(closes)
    assert values["bb_middle"] == pytest.approx(mean)
    assert values["bb_upper"] == pytest.approx(mean + 2 * std)
    assert values["bb_lower"] == pytest.approx(mean - 2 * std)
