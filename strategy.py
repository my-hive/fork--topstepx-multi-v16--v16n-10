import numpy as np
import pandas as pd

FAMILY = "fx-cad-rth-gap-reversal"
PARAMS = {"threshold": 0.003, "start": 34200, "end": 54000, "risk": 250}


def _rth_gap(d):
    rth = d[(d["ct_sec"] >= 30600) & (d["ct_sec"] < 54600)]
    bars = rth.groupby("session").agg(open=("open", "first"), close=("close", "last"))
    return bars["open"] / bars["close"].shift(1) - 1.0


def signals(panel):
    out = {}
    for symbol, d in panel.items():
        session = d["session"].astype(str).str[:10].to_numpy()
        sec = d["ct_sec"].to_numpy(int)
        gaps = _rth_gap(d)
        gaps.index = gaps.index.astype(str)
        gaps = gaps.to_dict()
        x = np.zeros(len(d), dtype=np.int64)
        direction = 0
        previous_session = None
        for i, sid in enumerate(session):
            if sid != previous_session:
                previous_session = sid
                gap = gaps.get(sid, np.nan)
                direction = -1 if gap > PARAMS["threshold"] else (1 if gap < -PARAMS["threshold"] else 0)
            active = direction if PARAMS["start"] <= sec[i] < PARAMS["end"] else 0
            x[i] = active
        out[symbol] = x
    return out


def entry_risk(panel):
    return {symbol: np.full(len(d), float(PARAMS["risk"])) for symbol, d in panel.items()}
