"""Causal equity-index lead-lag: ES impulse predicts delayed RTY direction."""
import numpy as np
import pandas as pd

FAMILY = "equity-lead-lag"
PARAMS = {"lookback": 5, "threshold": 0.00035, "stop_risk": 250, "entry_ct_sec": 34200, "exit_ct_sec": 54000}
LEAD = "ES"
TRADE = "RTY"

def signals(panel):
    out = {s: np.zeros(len(df), dtype=np.int64) for s, df in panel.items()}
    if LEAD not in panel or TRADE not in panel:
        return out
    lead, target = panel[LEAD], panel[TRADE]
    lr = lead.close.pct_change().rolling(PARAMS["lookback"]).sum().to_numpy()
    ts = target.ct_sec.to_numpy()
    active = 0
    for i in range(1, len(target)):
        if i and target.session.iloc[i] != target.session.iloc[i - 1]:
            active = 0
        if ts[i] >= PARAMS["exit_ct_sec"]:
            active = 0
        if active == 0 and PARAMS["entry_ct_sec"] <= ts[i] < PARAMS["exit_ct_sec"] and np.isfinite(lr[i]):
            if lr[i] > PARAMS["threshold"]: active = 1
            elif lr[i] < -PARAMS["threshold"]: active = -1
        out[TRADE][i] = active
    return out

def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["stop_risk"], float) if s == TRADE else np.zeros(len(df)) for s, df in panel.items()}
