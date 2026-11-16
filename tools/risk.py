"""Portfolio risk computed locally from the public price history.

Usage (from the repo root):
    python3 tools/risk.py holdings.csv > risk.json

holdings.csv columns: ticker,shares,currency   (avg_cost optional, ignored)
Positions stay on the machine running this; nothing is uploaded.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "history"
MEMORY = ["MU", "SNDK", "SKHY", "DRAM", "STX", "EWY"]


def r(x, n=2):
    try:
        x = float(x)
        return None if np.isnan(x) or np.isinf(x) else round(x, n)
    except Exception:
        return None


def close(t):
    p = HIST / f"{t.replace('^', '_').replace('.', '_')}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    return df["Close"].dropna()


h = pd.read_csv(sys.argv[1])
h["ticker"] = h["ticker"].str.strip()
prices = {t: close(t) for t in h["ticker"]}
missing = [t for t, c in prices.items() if c is None or c.empty]
h = h[~h["ticker"].isin(missing)].copy()
h["price"] = h["ticker"].map(lambda t: prices[t].iloc[-1])
us = h[h["currency"].str.upper() == "USD"].copy()
us["value_usd"] = us["shares"] * us["price"]
total = us["value_usd"].sum()
w = us.set_index("ticker")["value_usd"] / total
rets = pd.DataFrame({t: prices[t].pct_change() for t in w.index}).dropna().tail(126)
cov = rets.cov() * 252
vol = float(np.sqrt(w @ cov @ w))
mrc = (cov @ w) * w / vol
daily = rets @ w
out = {
    "as_of": str(rets.index[-1].date()),
    "us_value_usd": r(total, 0),
    "weights_pct": {t: r(100 * v, 1) for t, v in w.sort_values(ascending=False).items()},
    "risk_contribution_pct": {t: r(100 * v / vol, 1) for t, v in mrc.sort_values(ascending=False).items()},
    "portfolio_vol_ann_pct": r(100 * vol, 1),
    "var_95_1d_pct": r(-100 * np.percentile(daily, 5)),
    "expected_shortfall_95_1d_pct": r(-100 * daily[daily <= np.percentile(daily, 5)].mean()),
    "memory_cluster_pct": r(100 * w[[t for t in MEMORY if t in w]].sum(), 1),
    "max_single_name": [w.idxmax(), r(100 * w.max(), 1)],
}
for b in ("SPY", "SMH"):
    c = close(b)
    if c is not None:
        br = c.pct_change().reindex(daily.index)
        out[f"beta_vs_{b}"] = r(np.cov(daily, br)[0, 1] / br.var())
corr = rets.corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i + 1:]]
out["avg_pairwise_corr"] = r(np.mean([p[2] for p in pairs]))
out["most_correlated_pairs"] = [[a, b, r(c)] for a, b, c in sorted(pairs, key=lambda x: -x[2])[:6]]
iv = 1 / (rets.std() * np.sqrt(252))
out["inverse_vol_reference_weights_pct"] = {t: r(100 * v / iv.sum(), 1) for t, v in iv.sort_values(ascending=False).items()}
if missing:
    out["missing_history"] = missing
print(json.dumps(out, indent=2))
