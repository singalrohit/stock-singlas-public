"""Market-data pull for Rohit's portfolio signals.

Runs in GitHub Actions (Yahoo Finance is reachable there). Every section is
independent: one failing section or ticker never stops the others.

Writes into data/:
  latest.json              last close, daily change, 52w range, pre/post-market price if any
  metrics.json             precomputed technicals per holding and benchmark
  history/<T>.csv          5 years of daily OHLCV (holdings + benchmarks)
  events.json              upcoming earnings (with ET time) for holdings + upcoming FOMC days
  earnings_reactions.json  every past earnings: surprise %, 1/5/20-day moves, base-rate summary
  fomc_reactions.json      each holding's moves after past FOMC decisions, base-rate summary
  screen.json              momentum/trend screen of universe.txt, top candidates for new ideas
  status.json              run time, per-section counts and failures
"""
import gzip
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
HIST = DATA / "history"
HIST.mkdir(parents=True, exist_ok=True)
NY = "America/New_York"


def read_list(name):
    out = []
    for line in (ROOT / name).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        out += line.split()
    return out


def fname(t):
    return t.replace("^", "_").replace(".", "_")


def r(x, n=4):
    try:
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else round(x, n)
    except Exception:
        return None


def pct(a, b):
    return r((a / b - 1) * 100, 2) if b else None


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def ret(close, days):
    return pct(close.iloc[-1], close.iloc[-1 - days]) if len(close) > days else None


HOLDINGS = read_list("tickers.txt")
BENCH = read_list("benchmarks.txt")
UNIVERSE = [t for t in read_list("universe.txt") if t not in HOLDINGS]
FOMC = sorted(pd.to_datetime(read_list("fomc.txt")).date)
status = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sections": {}, "failed": []}
now_ny = pd.Timestamp.now(tz=NY)
SCREEN_POOL = []


def section(name):
    def deco(fn):
        try:
            res = fn()
            status["sections"][name] = res if res is not None else "ok"
        except Exception as e:
            status["sections"][name] = f"error: {e}"[:300]
            traceback.print_exc()
        return fn
    return deco


# ---------- 1. daily history, latest, metrics ----------
hist = {}
for t in HOLDINGS + BENCH:
    try:
        df = yf.Ticker(t).history(period="5y", interval="1d", auto_adjust=False)
        if df.empty:
            raise ValueError("no rows returned")
        df.index = df.index.tz_localize(None).normalize()
        df.index.name = "date"
        cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df]
        df = df[cols]
        # Right after the close Yahoo returns today's bar with a blank Close; drop it
        df = df[df["Close"].notna()]
        df.round(4).to_csv(HIST / f"{fname(t)}.csv", date_format="%Y-%m-%d")
        hist[t] = df
    except Exception as e:
        status["failed"].append({"ticker": t, "section": "history", "error": str(e)[:200]})

smh = hist.get("SMH", pd.DataFrame()).get("Close")
spy = hist.get("SPY", pd.DataFrame()).get("Close")


@section("latest_metrics")
def _latest():
    latest, metrics = {}, {}
    for t, df in hist.items():
        c = df["Close"]
        last, prev = c.iloc[-1], c.iloc[-2]
        yr = df.tail(252)
        entry = {
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
            "close": r(last), "prev_close": r(prev), "change_pct": pct(last, prev),
            "volume": int(df["Volume"].iloc[-1]) if "Volume" in df else None,
            "high_52w": r(yr["High"].max()), "low_52w": r(yr["Low"].min()),
        }
        # pre-/post-market: last 1-minute bar of the current session, if any
        try:
            m = yf.Ticker(t).history(period="2d", interval="1m", prepost=True)
            if not m.empty:
                ts = m.index[-1].tz_convert(NY)
                entry["extended_price"] = r(m["Close"].iloc[-1])
                entry["extended_time_et"] = ts.strftime("%Y-%m-%d %H:%M")
                entry["extended_change_pct"] = pct(m["Close"].iloc[-1], last)
        except Exception:
            pass
        latest[t] = entry

        sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
        daily = c.pct_change()
        mt = {
            "close": r(last), "sma50": r(sma50), "sma200": r(sma200),
            "above_sma50": bool(last > sma50) if not math.isnan(sma50) else None,
            "above_sma200": bool(last > sma200) if not math.isnan(sma200) else None,
            "golden_cross": bool(sma50 > sma200) if not math.isnan(sma200) else None,
            "rsi14": r(rsi(c).iloc[-1], 1),
            "ret_1m": ret(c, 21), "ret_3m": ret(c, 63), "ret_6m": ret(c, 126), "ret_12m": ret(c, 252),
            "vol20_ann_pct": r(daily.tail(20).std() * math.sqrt(252) * 100, 1),
            "daily_sigma_pct": r(daily.tail(20).std() * 100, 2),
            "drawdown_from_52w_high_pct": pct(last, yr["High"].max()),
        }
        for bname, b in (("smh", smh), ("spy", spy)):
            if b is not None and len(b) > 63 and t not in ("SMH", "SPY"):
                j = c.index.intersection(b.index)
                cc, bb = c.loc[j], b.loc[j]
                if len(j) > 63:
                    mt[f"rel_3m_vs_{bname}"] = r((cc.iloc[-1] / cc.iloc[-64] - bb.iloc[-1] / bb.iloc[-64]) * 100, 2)
        metrics[t] = mt
    (DATA / "latest.json").write_text(json.dumps(latest, indent=2))
    (DATA / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"tickers": len(latest)}


# ---------- 2. earnings: history of reactions + upcoming dates ----------
def reaction_day(df, ts):
    """Index position of the first session that reflects the report."""
    d = ts.tz_convert(NY) if ts.tzinfo else ts.tz_localize(NY)
    hour = d.hour + d.minute / 60
    day = pd.Timestamp(d.date())
    idx = df.index.searchsorted(day)
    if idx >= len(df):
        return None, None
    if hour >= 16:                     # after close -> next session
        timing = "AMC"
        if df.index[idx] == day:
            idx += 1
    elif 0 < hour < 9.5:               # before open -> same session
        timing = "BMO"
    else:                              # unknown time: take the bigger of the two candidate moves
        timing = "unknown"
        if idx + 1 < len(df) and idx >= 1:
            c = df["Close"]
            same = abs(c.iloc[idx] / c.iloc[idx - 1] - 1)
            nxt = abs(c.iloc[idx + 1] / c.iloc[idx] - 1)
            if nxt > same:
                idx += 1
    return (idx if idx < len(df) else None), timing


@section("earnings")
def _earnings():
    reactions, upcoming = {}, []
    for t in HOLDINGS:
        if t not in hist:
            continue
        df = hist[t]
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=28)
        except Exception as e:
            status["failed"].append({"ticker": t, "section": "earnings", "error": str(e)[:200]})
            continue
        if ed is None or ed.empty:
            continue
        rows = []
        for ts, row in ed.sort_index().iterrows():
            ts = pd.Timestamp(ts)
            ts_ny = ts.tz_convert(NY) if ts.tzinfo else ts.tz_localize(NY)
            reported = row.get("Reported EPS")
            if ts_ny > now_ny - timedelta(hours=6) and (reported is None or pd.isna(reported)):
                upcoming.append({
                    "ticker": t, "type": "earnings",
                    "datetime_et": ts_ny.strftime("%Y-%m-%d %H:%M"),
                    "timing": "AMC" if ts_ny.hour >= 16 else ("BMO" if 0 < ts_ny.hour < 9.5 else "time unknown"),
                    "eps_estimate": r(row.get("EPS Estimate")),
                })
                continue
            i, timing = reaction_day(df, ts_ny)
            if i is None or i < 1:
                continue
            c = df["Close"]
            base = c.iloc[i - 1]
            rec = {
                "date": ts_ny.strftime("%Y-%m-%d"), "timing": timing,
                "eps_est": r(row.get("EPS Estimate")), "eps_actual": r(reported),
                "surprise_pct": r(row.get("Surprise(%)"), 2),
                "gap_pct": pct(df["Open"].iloc[i], base),
                "d1_pct": pct(c.iloc[i], base),
                "d5_pct": pct(c.iloc[min(i + 4, len(c) - 1)], base) if i + 4 < len(c) else None,
                "d20_pct": pct(c.iloc[min(i + 19, len(c) - 1)], base) if i + 19 < len(c) else None,
            }
            # sector context: same window for SMH
            if smh is not None and df.index[i] in smh.index:
                j = smh.index.get_loc(df.index[i])
                if j >= 1:
                    rec["smh_d1_pct"] = pct(smh.iloc[j], smh.iloc[j - 1])
            rows.append(rec)
        d1 = [x["d1_pct"] for x in rows if x["d1_pct"] is not None]
        both = [x for x in rows if x["d1_pct"] is not None and x["d20_pct"] is not None]
        beats = [x for x in rows if (x["surprise_pct"] or 0) > 0 and x["d1_pct"] is not None]
        summary = {
            "n": len(d1),
            "avg_abs_d1_pct": r(np.mean(np.abs(d1)), 2) if d1 else None,
            "pct_up_d1": r(100 * np.mean([x > 0 for x in d1]), 0) if d1 else None,
            "beat_rate_pct": r(100 * np.mean([(x["surprise_pct"] or 0) > 0 for x in rows]), 0) if rows else None,
            "up_d1_after_beat_pct": r(100 * np.mean([x["d1_pct"] > 0 for x in beats]), 0) if beats else None,
            "fade_rate_pct": r(100 * np.mean([np.sign(x["d1_pct"]) != np.sign(x["d20_pct"]) for x in both]), 0) if both else None,
            "avg_d20_after_up_d1_pct": r(np.mean([x["d20_pct"] for x in both if x["d1_pct"] > 0]), 2) if any(x["d1_pct"] > 0 for x in both) else None,
            "avg_d20_after_down_d1_pct": r(np.mean([x["d20_pct"] for x in both if x["d1_pct"] < 0]), 2) if any(x["d1_pct"] < 0 for x in both) else None,
        }
        reactions[t] = {"summary": summary, "events": rows[-16:]}
    for d in FOMC:
        dt = pd.Timestamp(d).tz_localize(NY) + pd.Timedelta(hours=14)
        if dt > now_ny - timedelta(hours=6):
            upcoming.append({"ticker": "MACRO", "type": "FOMC decision", "datetime_et": dt.strftime("%Y-%m-%d %H:%M"),
                             "note": "Statement 14:00 ET, press conference 14:30 ET"})
    upcoming.sort(key=lambda x: x["datetime_et"])
    (DATA / "earnings_reactions.json").write_text(json.dumps(reactions, indent=2))
    (DATA / "events.json").write_text(json.dumps({
        "generated_utc": status["run_utc"],
        "note": "Earnings times come from Yahoo and can shift; confirm on the company IR page. Macro releases (CPI, jobs, PCE) are not listed here: check the BLS/BEA calendars.",
        "upcoming": upcoming}, indent=2))
    return {"tickers_with_history": len(reactions), "upcoming_events": len(upcoming)}


# ---------- 3. FOMC reactions ----------
@section("fomc")
def _fomc():
    out = {}
    past = [d for d in FOMC if pd.Timestamp(d) < pd.Timestamp(now_ny.date())]
    for t, df in hist.items():
        c = df["Close"]
        rows = []
        for d in past:
            day = pd.Timestamp(d)
            if day not in c.index:
                continue
            i = c.index.get_loc(day)
            if i < 1 or i + 5 >= len(c):
                continue
            base = c.iloc[i - 1]
            rows.append({"date": str(d), "d0_pct": pct(c.iloc[i], base),
                         "d1_pct": pct(c.iloc[i + 1], base), "d5_pct": pct(c.iloc[i + 5], base)})
        if not rows:
            continue
        d0 = [x["d0_pct"] for x in rows]
        out[t] = {"summary": {
            "n": len(rows),
            "avg_abs_d0_pct": r(np.mean(np.abs(d0)), 2),
            "pct_up_d0": r(100 * np.mean([x > 0 for x in d0]), 0),
            "fade_by_d5_pct": r(100 * np.mean([np.sign(x["d0_pct"]) != np.sign(x["d5_pct"]) for x in rows]), 0),
        }, "events": rows[-12:]}
    (DATA / "fomc_reactions.json").write_text(json.dumps(out, indent=2))
    return {"tickers": len(out), "past_meetings": len(past)}



# ---------- fundamentals helper (quality, valuation, revisions) ----------
def fundamentals(t):
    tk = yf.Ticker(t)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass
    g = lambda k: r(info.get(k))
    f = {
        "name": info.get("shortName"), "sector": info.get("sector"), "industry": info.get("industry"),
        "market_cap_bn": r((info.get("marketCap") or 0) / 1e9, 1) or None,
        "forward_pe": g("forwardPE"), "trailing_pe": g("trailingPE"), "peg": g("trailingPegRatio"),
        "ev_to_ebitda": g("enterpriseToEbitda"),
        "revenue_growth_pct": r((info.get("revenueGrowth") or float("nan")) * 100, 1),
        "earnings_growth_pct": r((info.get("earningsGrowth") or float("nan")) * 100, 1),
        "gross_margin_pct": r((info.get("grossMargins") or float("nan")) * 100, 1),
        "operating_margin_pct": r((info.get("operatingMargins") or float("nan")) * 100, 1),
        "roe_pct": r((info.get("returnOnEquity") or float("nan")) * 100, 1),
        "fcf_bn": r((info.get("freeCashflow") or float("nan")) / 1e9, 2),
        "fcf_yield_pct": r(100 * info["freeCashflow"] / info["marketCap"], 2) if info.get("freeCashflow") and info.get("marketCap") else None,
        "debt_to_equity": g("debtToEquity"),
        "beta": g("beta"),
        "analyst_target_mean": g("targetMeanPrice"), "analyst_rec_mean_1buy_5sell": g("recommendationMean"),
        "n_analysts": info.get("numberOfAnalystOpinions"),
    }
    # estimate revisions: change in current-FY and next-FY EPS consensus over 30 and 90 days
    try:
        et = tk.eps_trend
        if et is not None and not et.empty:
            for per, lab in (("0y", "fy0"), ("+1y", "fy1")):
                if per in et.index:
                    row = et.loc[per]
                    cur = row.get("current")
                    f[f"eps_{lab}_rev_30d_pct"] = pct(cur, row.get("30daysAgo")) if row.get("30daysAgo") else None
                    f[f"eps_{lab}_rev_90d_pct"] = pct(cur, row.get("90daysAgo")) if row.get("90daysAgo") else None
    except Exception:
        pass
    try:
        ev = tk.eps_revisions
        if ev is not None and not ev.empty and "0y" in ev.index:
            row = ev.loc["0y"]
            f["fy0_up_30d"] = int(row.get("upLast30days") or 0)
            f["fy0_down_30d"] = int(row.get("downLast30days") or 0)
    except Exception:
        pass
    # quality score 0-100 (profitability, cash generation, balance sheet, growth, revisions)
    pts, n = 0.0, 0
    def add(cond):
        nonlocal pts, n
        if cond is not None:
            n += 1
            pts += 1.0 if cond else 0.0
    add(f["operating_margin_pct"] > 10 if f["operating_margin_pct"] is not None else None)
    add(f["roe_pct"] > 12 if f["roe_pct"] is not None else None)
    add(f["fcf_bn"] > 0 if f["fcf_bn"] is not None else None)
    add(f["debt_to_equity"] < 150 if f["debt_to_equity"] is not None else None)
    add(f["revenue_growth_pct"] > 5 if f["revenue_growth_pct"] is not None else None)
    add(f["earnings_growth_pct"] > 0 if f["earnings_growth_pct"] is not None else None)
    rv = f.get("eps_fy1_rev_30d_pct")
    add(rv >= 0 if rv is not None else None)
    f["quality_score"] = r(100 * pts / n, 0) if n >= 4 else None
    f["quality_inputs"] = n
    return f

# ---------- 4. new-idea screen ----------
@section("screen")
def _screen():
    raw = yf.download(UNIVERSE, period="14mo", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    rows = []
    spy_c = spy
    for t in UNIVERSE:
        try:
            df = raw[t].dropna(how="all") if isinstance(raw.columns, pd.MultiIndex) else raw
            c = df["Close"].dropna()
            if len(c) < 260:
                continue
            last = c.iloc[-1]
            avg_dollar_vol = (c * df["Volume"]).tail(20).mean()
            if not t.endswith(".SI") and avg_dollar_vol < 50e6:
                continue
            sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
            hi = df["High"].tail(252).max()
            row = {
                "ticker": t, "close": r(last),
                "mom_12_1": pct(c.iloc[-22], c.iloc[-253]),   # 12-month return skipping last month
                "ret_6m": ret(c, 126), "ret_3m": ret(c, 63), "ret_1m": ret(c, 21),
                "trend_points": int(last > sma200) + int(sma50 > sma200) + int(last > sma50),
                "dist_from_52w_high_pct": pct(last, hi),
                "rsi14": r(rsi(c).iloc[-1], 1),
                "vol20_ann_pct": r(c.pct_change().tail(20).std() * math.sqrt(252) * 100, 1),
            }
            if spy_c is not None and not t.endswith(".SI"):
                j = c.index.intersection(spy_c.index)
                if len(j) > 64:
                    row["rel_3m_vs_spy"] = r((c.loc[j].iloc[-1] / c.loc[j].iloc[-64] - spy_c.loc[j].iloc[-1] / spy_c.loc[j].iloc[-64]) * 100, 2)
            rows.append(row)
        except Exception:
            continue
    if not rows:
        raise ValueError("no universe data")
    s = pd.DataFrame(rows)
    rank = lambda col: s[col].rank(pct=True).fillna(0.5)
    s["score"] = (100 * (0.30 * rank("mom_12_1") + 0.20 * rank("ret_6m") + 0.15 * rank("rel_3m_vs_spy" if "rel_3m_vs_spy" in s else "ret_3m")
                         + 0.20 * s["trend_points"] / 3 + 0.15 * (1 - (-s["dist_from_52w_high_pct"]).rank(pct=True)))).round(1)
    s["flags"] = s.apply(lambda x: ", ".join(f for f, ok in [
        ("extended RSI>75", (x["rsi14"] or 0) > 75),
        ("far from high", (x["dist_from_52w_high_pct"] or 0) < -20),
        ("below 200d", x["trend_points"] < 2),
        ("high vol", (x["vol20_ann_pct"] or 0) > 70)] if ok), axis=1)
    s = s.sort_values("score", ascending=False)
    s = s.rename(columns={"score": "momentum_score"})
    # enrich the 40 strongest momentum names with fundamentals, then rank on a multi-factor composite
    pool = s.head(40).to_dict("records")
    for row in pool:
        try:
            row.update(fundamentals(row["ticker"]))
        except Exception:
            pass
        q = row.get("quality_score")
        rev = row.get("eps_fy1_rev_30d_pct")
        rev_pts = 50 if rev is None else max(0, min(100, 50 + rev * 10))   # +5% revision -> 100
        val = row.get("forward_pe")
        val_pts = 50 if not val or val <= 0 else max(0, min(100, 100 - (val - 15) * 2))  # 15x -> 100, 65x -> 0
        row["composite_score"] = r(0.40 * row["momentum_score"] + 0.30 * (q if q is not None else 40)
                                   + 0.20 * rev_pts + 0.10 * val_pts, 1)
        row["passes_quality_bar"] = bool((q or 0) >= 70 and (row.get("fcf_bn") or 0) > 0
                                         and (rev is None or rev >= -1) and row["trend_points"] >= 2
                                         and (row.get("market_cap_bn") or 0) >= 10)
    pool.sort(key=lambda x: x["composite_score"] or 0, reverse=True)
    global SCREEN_POOL
    SCREEN_POOL = json.loads(pd.DataFrame(pool).to_json(orient="records"))
    top = pool[:25]
    for row in top[:15]:                                 # next earnings for the top candidates
        try:
            ed = yf.Ticker(row["ticker"]).get_earnings_dates(limit=4)
            fut = [pd.Timestamp(x) for x in ed.index if pd.Timestamp(x) > now_ny] if ed is not None else []
            row["next_earnings_et"] = min(fut).tz_convert(NY).strftime("%Y-%m-%d %H:%M") if fut else None
        except Exception:
            row["next_earnings_et"] = None
    clean = json.loads(pd.DataFrame(top).to_json(orient="records"))
    (DATA / "screen.json").write_text(json.dumps({
        "generated_utc": status["run_utc"], "universe_size": len(UNIVERSE), "scored": len(s),
        "method": "Two stages. momentum_score 0-100: 30% 12-1 momentum, 20% 6m return, 15% 3m relative strength vs SPY, 20% trend, 15% closeness to 52w high. The 40 best are enriched with fundamentals; composite_score = 40% momentum + 30% quality (margins, ROE, FCF, leverage, growth, revisions) + 20% EPS revisions (next FY, 30d) + 10% valuation (forward P/E). passes_quality_bar = quality>=70, FCF>0, revisions not falling, trend intact, market cap >= 10bn. Still verify news and earnings before any buy.",
        "top": clean}, indent=2))
    return {"scored": len(s)}


# ---------- 5. holdings fundamentals, implied earnings moves ----------
@section("holdings_fundamentals")
def _hfund():
    out = {}
    for t in HOLDINGS:
        try:
            out[t] = fundamentals(t)
        except Exception as e:
            status["failed"].append({"ticker": t, "section": "fundamentals", "error": str(e)[:200]})
    (DATA / "fundamentals.json").write_text(json.dumps(out, indent=2))
    return {"tickers": len(out)}


@section("implied_moves")
def _implied():
    """Options-implied move for holdings reporting within 21 days (ATM straddle / price)."""
    try:
        upcoming = json.loads((DATA / "events.json").read_text())["upcoming"]
    except Exception:
        upcoming = []
    out = {}
    for ev in upcoming:
        if ev.get("type") != "earnings" or ev["ticker"] not in hist or ev["ticker"].endswith(".SI"):
            continue
        when = pd.Timestamp(ev["datetime_et"]).tz_localize(NY)
        if when - now_ny > timedelta(days=21):
            continue
        t = ev["ticker"]
        try:
            tk = yf.Ticker(t)
            exps = [e for e in tk.options if pd.Timestamp(e).tz_localize(NY) + pd.Timedelta(hours=16) > when + pd.Timedelta(hours=18)]
            if not exps:
                continue
            exp = exps[0]
            ch = tk.option_chain(exp)
            px = hist[t]["Close"].iloc[-1]
            def mid(df):
                k = (df["strike"] - px).abs().idxmin()
                row = df.loc[k]
                b, a = row.get("bid") or 0, row.get("ask") or 0
                return (b + a) / 2 if b and a else row.get("lastPrice"), row["strike"], row.get("impliedVolatility")
            c, ks, civ = mid(ch.calls)
            p, _, piv = mid(ch.puts)
            out[t] = {"earnings_et": ev["datetime_et"], "expiry": exp, "strike": r(ks), "price": r(px),
                      "implied_move_pct": r(100 * (c + p) / px, 2),
                      "atm_iv_pct": r(100 * ((civ or 0) + (piv or 0)) / 2, 1)}
        except Exception as e:
            status["failed"].append({"ticker": t, "section": "implied_moves", "error": str(e)[:200]})
    (DATA / "implied_moves.json").write_text(json.dumps(out, indent=2))
    return {"tickers": len(out)}


# ---------- 6. portfolio risk and market regime ----------
REGIME = {}


@section("regime")
def _regime():
    # market regime (market-only, no holdings needed)
    for b in ("SPY", "QQQ", "SMH"):
        if b in hist:
            c = hist[b]["Close"]
            REGIME[b] = {"above_200d": bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1]),
                      "above_50d": bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1]), "ret_1m": ret(c, 21)}
    if "^VIX" in hist:
        v = hist["^VIX"]["Close"]
        REGIME["VIX"] = {"level": r(v.iloc[-1], 1), "pctile_1y": r(100 * (v.tail(252) < v.iloc[-1]).mean(), 0)}
    if "^TNX" in hist:
        y = hist["^TNX"]["Close"]
        REGIME["US10Y"] = {"level": r(y.iloc[-1], 2), "chg_1m_bp": r((y.iloc[-1] - y.iloc[-22]) * 100, 0) if y.iloc[-1] < 20 else r((y.iloc[-1] - y.iloc[-22]) * 10, 0)}
    up = sum(1 for b in ("SPY", "SMH") if REGIME.get(b, {}).get("above_200d"))
    vix_hi = REGIME.get("VIX", {}).get("level", 0) > 25
    REGIME["label"] = "risk-on" if up == 2 and not vix_hi else ("risk-off" if up == 0 or REGIME.get("VIX", {}).get("level", 0) > 30 else "mixed")
    (DATA / "regime.json").write_text(json.dumps(REGIME, indent=2, default=str))
    return {"label": REGIME["label"]}


@section("risk")
def _risk():
    if not (ROOT / "holdings.csv").exists():
        return "skipped (no holdings.csv: portfolio risk is computed privately)"
    h = pd.read_csv(ROOT / "holdings.csv")
    fx = 1.0
    try:
        fxdf = yf.Ticker("SGDUSD=X").history(period="5d")
        fx = float(fxdf["Close"].iloc[-1])
    except Exception:
        fx = 0.74
    h = h[h["ticker"].isin(hist.keys())].copy()
    h["price"] = h["ticker"].map(lambda t: hist[t]["Close"].iloc[-1])
    h["value_usd"] = h["shares"] * h["price"] * h["currency"].map(lambda c: fx if c == "SGD" else 1.0)
    us = h[h["currency"] == "USD"]
    total_us = us["value_usd"].sum()
    w = (us.set_index("ticker")["value_usd"] / total_us)
    rets = pd.DataFrame({t: hist[t]["Close"].pct_change() for t in w.index}).dropna().tail(126)
    cov = rets.cov() * 252
    port_vol = float(np.sqrt(w @ cov @ w))
    mrc = (cov @ w) * w / port_vol                     # contribution to portfolio volatility
    port_daily = rets @ w
    out = {
        "us_value_usd": r(total_us, 0), "sgd_usd": r(fx, 4),
        "weights_pct": {t: r(100 * v, 1) for t, v in w.sort_values(ascending=False).items()},
        "risk_contribution_pct": {t: r(100 * v / port_vol, 1) for t, v in mrc.sort_values(ascending=False).items()},
        "portfolio_vol_ann_pct": r(100 * port_vol, 1),
        "var_95_1d_pct": r(-100 * np.percentile(port_daily, 5), 2),
        "expected_shortfall_95_1d_pct": r(-100 * port_daily[port_daily <= np.percentile(port_daily, 5)].mean(), 2),
        "memory_cluster_pct": r(100 * w[[t for t in ["MU", "SNDK", "SKHY", "DRAM", "STX", "EWY"] if t in w]].sum(), 1),
        "max_single_name": [w.idxmax(), r(100 * w.max(), 1)],
    }
    for b in ("SPY", "SMH"):
        if b in hist:
            br = hist[b]["Close"].pct_change().reindex(port_daily.index)
            out[f"beta_vs_{b}"] = r(np.cov(port_daily, br)[0, 1] / br.var(), 2)
    corr = rets.corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i + 1:]]
    out["avg_pairwise_corr"] = r(np.mean([p[2] for p in pairs]), 2)
    out["most_correlated_pairs"] = [[a, b, r(c, 2)] for a, b, c in sorted(pairs, key=lambda x: -x[2])[:6]]
    # inverse-volatility weights: a risk-balanced reference, not a target
    iv = 1 / (rets.std() * np.sqrt(252))
    out["inverse_vol_reference_weights_pct"] = {t: r(100 * v / iv.sum(), 1) for t, v in iv.sort_values(ascending=False).items()}

    out["regime"] = REGIME
    (DATA / "risk.json").write_text(json.dumps(out, indent=2, default=str))
    return {"label": REGIME.get("label")}


# ---------- 7. track record of past signals ----------
@section("track_record")
def _track():
    p = ROOT / "signals" / "log.csv"
    if not p.exists():
        return {"signals": 0}
    log = pd.read_csv(p)
    if log.empty:
        (DATA / "track_record.json").write_text(json.dumps({"n": 0}, indent=2))
        return {"signals": 0}
    rows = []
    for _, x in log.iterrows():
        t = x["ticker"]
        if t not in hist:
            try:
                df = yf.Ticker(t).history(period="1y", auto_adjust=False)
                df.index = df.index.tz_localize(None).normalize()
                hist[t] = df
            except Exception:
                continue
        c = hist[t]["Close"]
        d = pd.Timestamp(str(x["date_et"])[:10])
        i = c.index.searchsorted(d)
        if i >= len(c):
            continue
        base = c.iloc[i]
        rec = {"date": str(x["date_et"])[:10], "ticker": t, "call": x["call"], "mode": x.get("mode")}
        for k in (5, 20, 60):
            if i + k < len(c):
                rec[f"ret_{k}d"] = pct(c.iloc[i + k], base)
                if smh is not None:
                    j = smh.index.searchsorted(c.index[i])
                    if j + k < len(smh):
                        rec[f"vs_smh_{k}d"] = r(rec[f"ret_{k}d"] - pct(smh.iloc[j + k], smh.iloc[j]), 2)
        rows.append(rec)
    df = pd.DataFrame(rows)
    summ = {}
    sign = {"Buy": 1, "Add partial": 1, "Add": 1, "NEW": 1, "Trim partial": -1, "Trim": -1, "Sell": -1}
    for call, g in df.groupby("call"):
        s = sign.get(call)
        d = {"n": len(g)}
        for k in (5, 20, 60):
            col = f"vs_smh_{k}d"
            if col in g and g[col].notna().any():
                v = g[col].dropna()
                d[f"avg_vs_smh_{k}d"] = r(v.mean(), 2)
                if s:
                    d[f"hit_rate_{k}d_pct"] = r(100 * ((v * s) > 0).mean(), 0)
        summ[call] = d
    (DATA / "track_record.json").write_text(json.dumps({"by_call": summ, "recent": rows[-40:]}, indent=2))
    return {"signals": len(rows)}


# ---------- 8. score my own forecasts ----------
@section("forecast_scoring")
def _fscore():
    p = ROOT / "signals" / "forecasts.csv"
    out_p = DATA / "forecast_scoring.json"
    empty = {"n_total": 0, "n_resolved": 0, "note": "No forecasts logged yet."}
    need = ["date_et", "ticker", "direction", "prob_pct", "window_days", "price"]
    f = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if f.empty or any(c not in f for c in need):
        out_p.write_text(json.dumps(empty, indent=2))
        return {"forecasts": 0}
    f = f.dropna(subset=need)
    cache, rows = {}, []
    for _, x in f.iterrows():
        t = str(x["ticker"]).strip()
        try:
            if t in hist:
                c = hist[t]["Close"]
            else:
                if t not in cache:
                    dfx = yf.Ticker(t).history(period="2y", auto_adjust=False)
                    dfx.index = dfx.index.tz_localize(None).normalize()
                    cache[t] = dfx["Close"]
                c = cache[t]
            d = pd.Timestamp(str(x["date_et"])[:10])
            i0 = int(c.index.searchsorted(d))
            w = int(x["window_days"])
            pre = str(x.get("asof", "pre")).strip().lower() == "pre"
            end = i0 + w - (1 if pre else 0)
            p_up = min(max(float(x["prob_pct"]) / 100, 0.5), 0.99)
            direction = str(x["direction"]).strip().lower()
            rec = {"date": str(d.date()), "ticker": t, "direction": direction, "prob": r(p_up, 2),
                   "window_days": w, "resolved": False}
            if i0 < len(c) and end < len(c):
                mv = float(c.iloc[end]) / float(x["price"]) - 1
                hit = (mv > 0) if direction == "up" else (mv < 0)
                rec.update(resolved=True, move_pct=r(mv * 100, 2), hit=bool(hit),
                           brier=r((p_up - float(hit)) ** 2, 4))
                inv = x.get("invalidation_price")
                if pd.notna(inv):
                    path = c.iloc[i0:end + 1]
                    rec["invalidation_hit"] = bool(path.min() <= float(inv)) if direction == "up" else bool(path.max() >= float(inv))
            rows.append(rec)
        except Exception as e:
            status["failed"].append({"ticker": t, "section": "forecast_scoring", "error": str(e)[:200]})
    res = [x for x in rows if x["resolved"]]
    out = {"n_total": len(rows), "n_resolved": len(res), "pending": len(rows) - len(res),
           "note": "Brier 0.25 is a coin flip; lower is better. Calibration compares stated odds with the share that came true."}
    if res:
        out["hit_rate_pct"] = r(100 * np.mean([x["hit"] for x in res]), 0)
        out["avg_stated_prob_pct"] = r(100 * np.mean([x["prob"] for x in res]), 0)
        out["brier"] = r(np.mean([x["brier"] for x in res]), 3)
        cal = []
        for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
            g = [x for x in res if lo <= x["prob"] < hi]
            if g:
                cal.append({"stated_range": f"{int(lo * 100)}-{min(int(hi * 100), 99)}%", "n": len(g),
                            "avg_stated_pct": r(100 * np.mean([x["prob"] for x in g]), 0),
                            "came_true_pct": r(100 * np.mean([x["hit"] for x in g]), 0)})
        out["calibration"] = cal
        by_dir = {}
        for dname in ("up", "down"):
            g = [x for x in res if x["direction"] == dname]
            if g:
                by_dir[dname] = {"n": len(g), "hit_rate_pct": r(100 * np.mean([x["hit"] for x in g]), 0)}
        out["by_direction"] = by_dir
    out["recent"] = rows[-30:]
    out_p.write_text(json.dumps(out, indent=2))
    return {"forecasts": len(rows), "resolved": len(res)}


# ---------- 9. point-in-time snapshots for future backtests ----------
@section("universe_snapshot")
def _usnap():
    day = now_ny.strftime("%Y-%m-%d")
    if not (os.environ.get("FULL_SNAPSHOT") == "1" or (now_ny.weekday() == 4 and now_ny.hour < 12)):
        return "skipped (runs on Friday pre-market runs)"
    t0, out = time.time(), {}
    for t in UNIVERSE:
        if time.time() - t0 > 480:
            break
        try:
            out[t] = fundamentals(t)
        except Exception:
            pass
    snap = DATA / "snapshots"
    snap.mkdir(exist_ok=True)
    with gzip.open(snap / f"universe_{day}.json.gz", "wt") as g:
        json.dump({"run_utc": status["run_utc"], "fundamentals": out}, g)
    return {"tickers": len(out), "seconds": int(time.time() - t0)}


@section("snapshot")
def _snap():
    def load(name):
        q = DATA / name
        return json.loads(q.read_text()) if q.exists() else None
    tag = "pre" if now_ny.hour < 12 else "post"
    day = now_ny.strftime("%Y-%m-%d")
    snap = DATA / "snapshots"
    snap.mkdir(exist_ok=True)
    risk = load("risk.json") or {}
    body = {"run_utc": status["run_utc"], "session": tag,
            "holdings_fundamentals": load("fundamentals.json"), "metrics": load("metrics.json"),
            "implied_moves": load("implied_moves.json"), "regime": risk.get("regime"),
            "risk_summary": {k: risk.get(k) for k in ("us_value_usd", "portfolio_vol_ann_pct", "var_95_1d_pct",
                                                      "beta_vs_SPY", "beta_vs_SMH", "memory_cluster_pct")},
            "screen_pool": SCREEN_POOL}
    with gzip.open(snap / f"{day}_{tag}.json.gz", "wt") as g:
        json.dump(body, g)
    return {"file": f"{day}_{tag}"}


ok = len(hist)
status["ok"] = ok
status["holdings_ok"] = sum(t in hist for t in HOLDINGS)
(DATA / "status.json").write_text(json.dumps(status, indent=2))
print(json.dumps(status, indent=2))
sys.exit(0 if ok else 1)
