"""Turn-of-month calendar drift, ATR dollar-stop sizing.
Long MNQ and MYM for the first 5 trading sessions of each month, cash session 09:30-15:00 CT.
Trailing 10-session cash-range ATR per contract defines causal dollar risk;
size = floor(min(300, risk_cap) / dollar_risk). Stop triggers intraday on cash low.
"""
import numpy as np
import pandas as pd

FAMILY = "equity-tom-monthly-drift"
PARAMS = {"n_first_sessions": 5, "atr_sessions": 10, "atr_mult": 1.0, "risk_cap": 300,
          "entry_ct_sec": 34200, "exit_ct_sec": 54000}
BASKET = ("MNQ", "MYM")


def _session_metrics(df):
    sess = df.session.astype(str).to_numpy()
    sec = df.ct_sec.to_numpy()
    high = df.high.to_numpy(dtype=float)
    low = df.low.to_numpy(dtype=float)
    rng = {}
    for label in pd.unique(sess):
        ix = np.flatnonzero(sess == label)
        cash = ix[(sec[ix] >= 30600) & (sec[ix] < PARAMS["exit_ct_sec"])]
        if len(cash) > 60:
            rng[label] = high[cash].max() - low[cash].min()
    labels = list(rng)
    return labels, rng


def entry_risk(panel):
    out = {}
    for symbol, df in panel.items():
        risk = np.zeros(len(df), dtype=float)
        if symbol in BASKET:
            labels, rng = _session_metrics(df)
            known = {}
            sess = df.session.astype(str).to_numpy()
            for i, label in enumerate(labels):
                prior = [rng[x] for x in labels[max(0, i - 15):i]]
                if len(prior) >= 3:
                    known[label] = PARAMS["atr_mult"] * float(np.mean(prior[-PARAMS["atr_sessions"]:]))
            risk = np.array([known.get(s, 0.0) for s in sess])
        out[symbol] = risk
    return out


def signals(panel):
    targets = {}
    for symbol, df in panel.items():
        signal = np.zeros(len(df), dtype=np.int64)
        targets[symbol] = signal
        if symbol not in BASKET:
            continue
        sess = df.session.astype(str).to_numpy()
        sec = df.ct_sec.to_numpy()
        opened = df.open.to_numpy(dtype=float)
        low = df.low.to_numpy(dtype=float)
        unique = pd.unique(sess)
        months = pd.to_datetime(unique).to_period("M")
        ranks = pd.Series(range(len(unique)), index=unique).groupby(months).rank(method="first").to_dict()
        labels, rng = _session_metrics(df)
        atr = {}
        for i, label in enumerate(labels):
            prior = [rng[x] for x in labels[max(0, i - 15):i]]
            if len(prior) >= 3:
                atr[label] = PARAMS["atr_mult"] * float(np.mean(prior[-PARAMS["atr_sessions"]:]))
        for label in unique:
            if ranks.get(label, 1e9) > PARAMS["n_first_sessions"]:
                continue
            stop = atr.get(label)
            if not stop:
                continue
            qty = int(min(50, PARAMS["risk_cap"] // stop))
            if qty < 1:
                continue
            ix = np.flatnonzero((sess == label) & (sec >= PARAMS["entry_ct_sec"]) & (sec < PARAMS["exit_ct_sec"]))
            if len(ix) < 30:
                continue
            entry = opened[ix[0]]
            active = True
            for j in ix:
                if active and low[j] <= entry - stop:
                    active = False
                signal[j] = qty if active else 0
        targets[symbol] = signal
    return targets
