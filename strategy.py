"""Turn-of-month calendar drift on equity index micro contracts.
Go long MNQ and MYM for the first 5 trading sessions of each month, flat otherwise.
Trades during cash session 09:30-15:00 CT; no entries between 08:30-09:30 CT.
Calendar month and session rank are strictly causal.
"""
import numpy as np
import pandas as pd

FAMILY = "equity-tom-monthly-drift"
PARAMS = {"n_first_sessions": 5, "entry_ct_sec": 34200, "exit_ct_sec": 54000, "stop_dollars": 200}
BASKET = ("MNQ", "MYM")


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["stop_dollars"], dtype=float) for s, df in panel.items()}


def signals(panel):
    targets = {}
    for symbol, df in panel.items():
        signal = np.zeros(len(df), dtype=np.int64)
        targets[symbol] = signal
        if symbol not in BASKET:
            continue
        session_str = df.session.astype(str).to_numpy()
        sec = df.ct_sec.to_numpy()
        unique = pd.unique(session_str)
        months = pd.to_datetime(unique).to_period("M")
        ranks = pd.Series(range(len(unique)), index=unique).groupby(months).rank(method="first").to_dict()
        eligible = {s for s, r in ranks.items() if r <= PARAMS["n_first_sessions"]}
        mask = np.isin(session_str, list(eligible)) & (sec >= PARAMS["entry_ct_sec"]) & (sec < PARAMS["exit_ct_sec"])
        signal[mask] = 1
    return targets
