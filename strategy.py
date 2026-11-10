import numpy as np

FAMILY = "cross-sectional-rth-trend"
PARAMS = {"lookback": 60, "threshold": 0.001, "entry_start": 34200, "entry_end": 54000, "risk": 250}


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    for symbol, df in panel.items():
        session = df["session"].astype(str).to_numpy()
        close = df["close"].to_numpy(float)
        sec = df["ct_sec"].to_numpy(int)
        days = {}
        for i, day in enumerate(session):
            days.setdefault(day, []).append(i)
        ordered = sorted(days)
        for pos, day in enumerate(ordered):
            ix = days[day]
            # prior completed session RTH 60-minute return, causal
            prev = days[ordered[pos - 1]] if pos else []
            if len(prev) < PARAMS["lookback"]:
                continue
            r = close[prev[-1]] / close[prev[-PARAMS["lookback"]]] - 1.0
            if abs(r) < PARAMS["threshold"]:
                continue
            side = 1 if r > 0 else -1
            for i in ix:
                if PARAMS["entry_start"] <= sec[i] < PARAMS["entry_end"]:
                    out[symbol][i] = side
    return out


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["risk"], dtype=float) for s, d in panel.items()}
