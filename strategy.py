import numpy as np

FAMILY = "rates-opening-range-breakout"
PARAMS = {"range_end": 34200, "entry_end": 39600, "exit": 52200, "min_range": 0.0004, "risk": 250}


def signals(panel):
    out = {s: np.zeros(len(df), dtype=np.int64) for s, df in panel.items()}
    for symbol, df in panel.items():
        close = df["close"].to_numpy(float)
        high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float)
        sec = df["ct_sec"].to_numpy()
        session = df["session"].to_numpy()
        side = np.zeros(len(df), dtype=np.int64)
        hi = lo = np.nan
        active = 0
        for i in range(len(df)):
            new = i == 0 or session[i] != session[i - 1]
            if new:
                hi, lo, active = np.nan, np.nan, 0
            if sec[i] < PARAMS["range_end"]:
                hi = high[i] if not np.isfinite(hi) else max(hi, high[i])
                lo = low[i] if not np.isfinite(lo) else min(lo, low[i])
            if sec[i] >= PARAMS["exit"]:
                active = 0
            if (not new and active == 0 and sec[i] > PARAMS["range_end"] and
                    sec[i] <= PARAMS["entry_end"] and np.isfinite(hi) and
                    (hi - lo) / close[i] >= PARAMS["min_range"]):
                if close[i] > hi:
                    active = 1
                elif close[i] < lo:
                    active = -1
            side[i] = active
        out[symbol] = side
    return out


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk"], float) for s, df in panel.items()}
