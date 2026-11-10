import numpy as np

FAMILY = "equity-overnight-gap-fade"
PARAMS = {"gap_lookback": 60, "gap_quantile": 0.80, "entry_start": 34200, "entry_end": 54000, "risk": 250}


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "MYM" not in panel:
        return out
    df = panel["MYM"]
    sess = df["session"].astype(str).to_numpy()
    sec = df["ct_sec"].to_numpy(int)
    op = df["open"].to_numpy(float)
    close = df["close"].to_numpy(float)
    days = {}
    for i, day in enumerate(sess):
        days.setdefault(day, []).append(i)
    ordered = sorted(days)
    closes = []
    gaps = []
    for pos, day in enumerate(ordered):
        ix = [i for i in days[day] if sec[i] >= 30600 and sec[i] < 54000]
        if not ix:
            continue
        rth_open = op[ix[0]]
        if closes:
            gap = rth_open / closes[-1] - 1.0
            gaps.append(gap)
            hist = np.abs(np.asarray(gaps[-PARAMS["gap_lookback"]:]))
            if len(hist) >= PARAMS["gap_lookback"] // 2 and abs(gap) >= np.quantile(hist, PARAMS["gap_quantile"]):
                side = -1 if gap > 0 else 1
                for i in ix:
                    if PARAMS["entry_start"] <= sec[i] < PARAMS["entry_end"]:
                        out["MYM"][i] = side
        closes.append(close[ix[-1]])
    return out


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["risk"], dtype=float) for s, d in panel.items()}
