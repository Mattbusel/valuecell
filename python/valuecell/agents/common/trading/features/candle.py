from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fin_primitives import ta as fp_ta

from valuecell.agents.common.trading.constants import (
    FEATURE_GROUP_BY_INTERVAL_PREFIX,
    FEATURE_GROUP_BY_KEY,
)
from valuecell.agents.common.trading.models import Candle, FeatureVector

from .interfaces import CandleBasedFeatureComputer


class SimpleCandleFeatureComputer(CandleBasedFeatureComputer):
    """Computes basic momentum and volume features."""

    def compute_features(
        self,
        candles: Optional[List[Candle]] = None,
        meta: Optional[Dict[str, object]] = None,
    ) -> List[FeatureVector]:
        if not candles:
            return []

        grouped: Dict[str, List[Candle]] = defaultdict(list)
        for candle in candles:
            grouped[candle.instrument.symbol].append(candle)

        features: List[FeatureVector] = []
        for symbol, series in grouped.items():
            # Build a DataFrame for indicator calculations
            series.sort(key=lambda item: item.ts)
            rows = [
                {
                    "ts": c.ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "interval": c.interval,
                }
                for c in series
            ]
            df = pd.DataFrame(rows)

            # EMAs
            df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

            # MACD
            df["macd"] = df["ema_12"] - df["ema_26"]
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

            # RSI (Wilder smoothing), same values as pandas-ta / TA-Lib
            df["rsi"] = fp_ta.rsi(df["close"], length=14)

            # Bollinger Bands (population std), same values as pandas-ta / TA-Lib
            bbands = fp_ta.bbands(df["close"], length=20, std=2)
            df["bb_lower"] = bbands["BBL_20_2.0"]
            df["bb_middle"] = bbands["BBM_20_2.0"]
            df["bb_upper"] = bbands["BBU_20_2.0"]

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            change_pct = (
                (float(last.close) - float(prev.close)) / float(prev.close)
                if prev.close
                else 0.0
            )

            values = {
                "close": float(last.close),
                "volume": float(last.volume),
                "change_pct": float(change_pct),
                "ema_12": (
                    float(last.get("ema_12", np.nan))
                    if not pd.isna(last.get("ema_12"))
                    else None
                ),
                "ema_26": (
                    float(last.get("ema_26", np.nan))
                    if not pd.isna(last.get("ema_26"))
                    else None
                ),
                "ema_50": (
                    float(last.get("ema_50", np.nan))
                    if not pd.isna(last.get("ema_50"))
                    else None
                ),
                "macd": (
                    float(last.get("macd", np.nan))
                    if not pd.isna(last.get("macd"))
                    else None
                ),
                "macd_signal": (
                    float(last.get("macd_signal", np.nan))
                    if not pd.isna(last.get("macd_signal"))
                    else None
                ),
                "macd_histogram": (
                    float(last.get("macd_histogram", np.nan))
                    if not pd.isna(last.get("macd_histogram"))
                    else None
                ),
                "rsi": (
                    float(last.get("rsi", np.nan))
                    if not pd.isna(last.get("rsi"))
                    else None
                ),
                "bb_upper": (
                    float(last.get("bb_upper", np.nan))
                    if not pd.isna(last.get("bb_upper"))
                    else None
                ),
                "bb_middle": (
                    float(last.get("bb_middle", np.nan))
                    if not pd.isna(last.get("bb_middle"))
                    else None
                ),
                "bb_lower": (
                    float(last.get("bb_lower", np.nan))
                    if not pd.isna(last.get("bb_lower"))
                    else None
                ),
            }

            # Build feature meta
            window_start_ts = int(rows[0]["ts"]) if rows else int(last["ts"])
            window_end_ts = int(last["ts"])
            interval = series[-1].interval
            fv_meta = {
                FEATURE_GROUP_BY_KEY: f"{FEATURE_GROUP_BY_INTERVAL_PREFIX}{interval}",
                "interval": interval,
                "count": len(series),
                "window_start_ts": window_start_ts,
                "window_end_ts": window_end_ts,
            }
            if meta:
                # Merge provided meta (doesn't overwrite core keys unless intended)
                for k, v in meta.items():
                    fv_meta.setdefault(k, v)

            features.append(
                FeatureVector(
                    ts=int(last["ts"]),
                    instrument=series[-1].instrument,
                    values=values,
                    meta=fv_meta,
                )
            )
        return features
