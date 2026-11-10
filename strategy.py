import numpy as np

FAMILY = "equity-month-end-seasonality"
PARAMS = {"entry_start": 34200, "entry_end": 54000, "risk": 250, "days": 3}


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "MNQ" not in panel:
        return out
    df = panel["MNQ"]
    sess = df["session"].astype(str).to_numpy()
    sec = df["ct_sec"].to_numpy(int)
    days = {}
    for i, day in enumerate(sess):
        days.setdefault(day, []).append(i)
    ordered = sorted(days)
    for pos, day in enumerate(ordered):
        month = day[:7]
        future = ordered[min(pos + PARAMS["days"], len(ordered) - 1)][:7]
        if future == month:
            continue
        for i in days[day]:
            if PARAMS["entry_start"] <= sec[i] < PARAMS["entry_end"]:
                out["MNQ"][i] = -1
    return out


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["risk"], dtype=float) for s, d in panel.items()}
