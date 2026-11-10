import numpy as np

FAMILY = "equity-weekday-seasonality"
PARAMS = {"weekday": 4, "entry_start": 36000, "entry_end": 54000, "risk": 250}


def signals(panel):
    out = {s: np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}
    if "MNQ" not in panel:
        return out
    df = panel["MNQ"]
    sec = df["ct_sec"].to_numpy(int)
    day = df["session"].dt.dayofweek.to_numpy()
    active = (day == PARAMS["weekday"]) & (sec >= PARAMS["entry_start"]) & (sec < PARAMS["entry_end"])
    out["MNQ"][active] = -1
    return out


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["risk"], dtype=float) for s, d in panel.items()}
