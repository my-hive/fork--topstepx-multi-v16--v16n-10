import numpy as np

FAMILY = "fx-london-ny-handoff-breakout"
PARAMS = {"range_minutes": 30, "buffer": 0.0002, "risk": 250}


def signals(panel):
    out = {}
    for symbol, d in panel.items():
        n = len(d)
        sec = d["ct_sec"].to_numpy(dtype=int)
        sessions = d["session"].astype(str).to_numpy()
        high = d["high"].to_numpy(float)
        low = d["low"].to_numpy(float)
        close = d["close"].to_numpy(float)
        target = np.zeros(n, dtype=np.int64)
        start = {}
        end = {}
        for i, sid in enumerate(sessions):
            if 30600 <= sec[i] < 30600 + PARAMS["range_minutes"] * 60:
                start[sid] = max(start.get(sid, -np.inf), high[i])
                end[sid] = min(end.get(sid, np.inf), low[i])
            if sid in start and sec[i] >= 32400 and sec[i] < 54000:
                if close[i] > start[sid] + PARAMS["buffer"]:
                    target[i] = 1
                elif close[i] < end[sid] - PARAMS["buffer"]:
                    target[i] = -1
        out[symbol] = target
    return out


def entry_risk(panel):
    return {symbol: np.full(len(d), PARAMS["risk"], dtype=float) for symbol, d in panel.items()}
