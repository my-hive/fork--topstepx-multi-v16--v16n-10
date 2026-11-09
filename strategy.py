"""Causal session-opening gap reversion across FX and rates."""
import numpy as np
import pandas as pd
FAMILY = "prior-close-gap-reversion-multiasset"
PARAMS = {"lookback": 30, "gap": 0.0005, "start": 34200, "end": 37800, "risk": 240}
def signals(panel):
    out={s:np.zeros(len(d),dtype=np.int64) for s,d in panel.items()}
    for s,d in panel.items():
        sec=d.ct_sec.to_numpy(int); sess=d.session.astype(str).to_numpy(); c=d.close.to_numpy(float)
        r=pd.Series(c).pct_change(PARAMS["lookback"]).to_numpy(); x=np.zeros(len(d),dtype=np.int64); prev=""; active=0
        for i in range(len(d)):
            if sess[i]!=prev: active=0; prev=sess[i]
            if sec[i]>=PARAMS["end"]: active=0
            elif active==0 and PARAMS["start"]<=sec[i]<PARAMS["end"] and np.isfinite(r[i]):
                if r[i]>PARAMS["gap"]: active=-1
                elif r[i]<-PARAMS["gap"]: active=1
            x[i]=active
        out[s]=x
    return out
def entry_risk(panel): return {s:np.full(len(d),float(PARAMS["risk"])) for s,d in panel.items()}
