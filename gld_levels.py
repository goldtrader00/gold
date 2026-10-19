#!/usr/bin/env python3
"""
gld_levels.py - CBOE GLD options + CFTC COT -> XAUUSD levels dashboard

Install:  py -m pip install requests pandas numpy scipy yfinance
Run:      py gld_levels.py --spot 4151 --max-dte 14 --vwap-week 4132 --vwap-month 4098
Output:   output/levels.json, output/dashboard.html, snapshots/<date>.csv (daily OI history)

What each piece is (and how much to trust it):
- Call/Put OI walls, max pain, GEX, gamma flip: computed from CBOE GLD options (delayed ~15 min).
- Put wall / Call wall: biggest put OI strike below price / biggest call OI strike above price.
- New CALL / New PUT: OI increase vs the last earlier snapshot whose OI differs. Needs one prior run
  on a different day. OI change shows positions opened, NOT direction or who is long/short.
- VWAP W / VWAP M: type them in from TradingView (--vwap-week / --vwap-month), or use --paxg to
  compute them from Binance PAXGUSDT (24/7 gold token, shifted to spot by its current premium).
- Funds BUY / SELL: MY ESTIMATE, not real fund orders. For each recent COT week where Managed Money
  net position changed by >= --funds-min contracts, the zone is that week's volume-weighted gold
  price (COMEX futures from yfinance, shifted to spot), snapped to a GLD strike zone.
  Notional = |change| x 100 oz x price.
- Strike->XAUUSD uses ratio = spot / GLD price unless you pass --ratio.
- GEX assumes dealers long calls / short puts (a convention, not observed).
"""
import argparse
import html
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
COT_URL = "https://publicreporting.cftc.gov/resource/{ds}.json"
COT_DATASET = "72hh-3qpy"   # CFTC Disaggregated, futures only (verify on publicreporting.cftc.gov)
GOLD_CODE = "088691"        # COMEX gold
OCC = re.compile(r"^(?P<root>[A-Z]+\d?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")
DEFAULT_RATIO = 10.88   # ~ gold price per GLD share (1 / oz-of-gold-per-share); drifts UP ~0.4%/yr as fees accrue
RISK_FREE = 0.04
CONTRACT_MULT = 100
KEY = ["expiry_s", "strike", "is_call"]


# ----------------------------------------------------------------- data pull
def fetch_chain(symbol, from_file=None):
    if from_file:
        raw = json.loads(Path(from_file).read_text())
    else:
        r = requests.get(CBOE_URL.format(sym=symbol),
                         headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        raw = r.json()
    data = raw["data"]
    return float(data["current_price"]), pd.DataFrame(data["options"]), data.get("last_trade_time")


def _nth_sunday(y, m, n):
    d = datetime(y, m, 1)
    return d + timedelta(days=(6 - d.weekday()) % 7 + 7 * (n - 1))


def et_dst(t_et):
    return _nth_sunday(t_et.year, 3, 2) <= t_et < _nth_sunday(t_et.year, 11, 1)


def utc_to_et(now_utc):
    t = now_utc.replace(tzinfo=None)
    return t - timedelta(hours=4 if et_dst(t - timedelta(hours=4)) else 5)


def quote_utc(qtime):
    """CBOE quote time is Eastern without a zone -> naive UTC datetime (or None)."""
    try:
        t = datetime.fromisoformat(str(qtime))
        return t + timedelta(hours=4 if et_dst(t) else 5)
    except Exception:
        return None


def quote_age_min(qtime, now):
    q = quote_utc(qtime)
    return max(0.0, (now.replace(tzinfo=None) - q).total_seconds() / 60) if q else None


SPOT_SOURCE = "manual"


def _spot_swissquote():
    r = requests.get("https://forex-data-feed.swissquote.com/public-quote-feed/XAU/USD", timeout=10)
    r.raise_for_status()
    p = r.json()[0]["spreadProfilePrices"][0]
    return (float(p["bid"]) + float(p["ask"])) / 2


def _spot_yfinance():
    import yfinance as yf
    h = yf.Ticker("XAUUSD=X").history(period="1d", interval="1m")
    return float(h["Close"].iloc[-1])


def get_spot_gold(manual):
    """Spot gold: --spot if given, else the first live source that agrees with PAXG to within $60 (sanity check)."""
    global SPOT_SOURCE
    if manual:
        SPOT_SOURCE = "manual"
        return float(manual)
    paxg_last = None
    try:
        k = fetch_klines("PAXGUSDT", int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)).timestamp() * 1000), "5m")
        paxg_last = float(k["close"].iloc[-1])
    except Exception as e:
        print(f"PAXG check failed: {e}", file=sys.stderr)
    for name, fn in (("Swissquote", _spot_swissquote), ("Yahoo XAUUSD=X", _spot_yfinance)):
        try:
            px = fn()
        except Exception as e:
            print(f"{name} failed: {e}", file=sys.stderr)
            continue
        if paxg_last and abs(px - paxg_last) > 60:
            print(f"{name} gave {px:.2f}, far from PAXG {paxg_last:.2f}: ignored", file=sys.stderr)
            continue
        SPOT_SOURCE = name
        print(f"Spot gold from {name}: {px:.2f}")
        return px
    if paxg_last:
        SPOT_SOURCE = "PAXG (no real gold price)"
        print(f"WARNING: spot taken from PAXG ({paxg_last:.2f}); it can sit $1-20 away from OANDA.")
        return paxg_last
    sys.exit("Could not get spot gold. Re-run with --spot <price>.")


def parse(df):
    m = df["option"].str.extract(OCC)
    df = pd.concat([df, m], axis=1).dropna(subset=["ymd"]).copy()
    df["expiry"] = pd.to_datetime(df["ymd"], format="%y%m%d")
    df["expiry_s"] = df["expiry"].dt.strftime("%Y-%m-%d")
    df["strike"] = df["k"].astype(float) / 1000.0
    df["is_call"] = df["cp"] == "C"
    for c in ("open_interest", "volume", "iv", "gamma", "bid", "ask"):
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0.0)
    return df


def prepare(df, today, max_dte, band, gld_px):
    df = df.copy()
    df["dte"] = (df["expiry"] - today).dt.days
    df = df[(df["dte"] >= 0) & (df["dte"] <= max_dte)]
    df = df[(df["strike"] >= gld_px * (1 - band)) & (df["strike"] <= gld_px * (1 + band))]
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    return df.reset_index(drop=True)


# ------------------------------------------------------------ OI snapshots
def find_delta(snap_dir, today_s, cur):
    """OI change vs the most recent EARLIER snapshot in which any OI differs.
    Returns (snapshot_date, deltas, earlier_snapshot_existed)."""
    seen = False
    for f in sorted(Path(snap_dir).glob("*.csv"), reverse=True):
        if f.stem >= today_s:
            continue
        seen = True
        prev = pd.read_csv(f).rename(columns={"open_interest": "prev_oi"})
        m = cur.merge(prev, on=KEY, how="left")
        d = (m["open_interest"] - m["prev_oi"].fillna(0)).to_numpy()
        if np.any(np.abs(d) > 0):
            return f.stem, d, True
    return None, None, seen


# -------------------------------------------------------------------- maths
def bs_gamma(S, K, T, iv, r=RISK_FREE):
    iv = np.maximum(iv, 1e-4)
    T = np.maximum(T, 1e-6)
    d1 = (np.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return norm.pdf(d1) / (S * iv * np.sqrt(T))


def net_gex(df, S):
    T = np.maximum(df["dte"].to_numpy(), 0.25) / 365.0
    g = bs_gamma(S, df["strike"].to_numpy(), T, df["iv"].to_numpy())
    sign = np.where(df["is_call"].to_numpy(), 1.0, -1.0)
    return float(np.sum(sign * g * df["open_interest"].to_numpy() * CONTRACT_MULT * S * S * 0.01))


def gamma_flip(df, S):
    grid = np.linspace(S * 0.85, S * 1.15, 301)
    vals = np.array([net_gex(df, s) for s in grid])
    flips = []
    for i in range(len(grid) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            flips.append(float(grid[i] - vals[i] * (grid[i + 1] - grid[i]) / (vals[i + 1] - vals[i])))
    return min(flips, key=lambda x: abs(x - S)) if flips else None


def max_pain(df):
    strikes = np.sort(df["strike"].unique())
    if len(strikes) == 0:
        return None
    c = df[df["is_call"]].groupby("strike")["open_interest"].sum().reindex(strikes, fill_value=0).to_numpy()
    p = df[~df["is_call"]].groupby("strike")["open_interest"].sum().reindex(strikes, fill_value=0).to_numpy()
    pain = [float(np.sum(c * np.maximum(k - strikes, 0)) + np.sum(p * np.maximum(strikes - k, 0))) for k in strikes]
    return float(strikes[int(np.argmin(pain))])


# ---------------------------------------------------------------- COT/funds
def fetch_cot(dataset=COT_DATASET, weeks=70):
    r = requests.get(COT_URL.format(ds=dataset), timeout=30, params={
        "$where": f"cftc_contract_market_code='{GOLD_CODE}'",
        "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": weeks})
    r.raise_for_status()
    d = pd.DataFrame(r.json())
    if d.empty:
        raise ValueError("empty COT response")

    def col(prefix, side):
        c = [x for x in d.columns if x.startswith(prefix) and f"{side}_all" in x and "spread" not in x
             and not x.startswith(("change", "pct", "conc"))]
        if not c:
            raise KeyError(f"COT column {prefix}/{side} not found; columns: {list(d.columns)[:12]}...")
        return pd.to_numeric(d[c[0]], errors="coerce")

    out = pd.DataFrame({
        "date": pd.to_datetime(d["report_date_as_yyyy_mm_dd"]),
        "mm_net": col("m_money", "long") - col("m_money", "short"),
        "swap_net": col("swap", "long") - col("swap", "short"),
    })
    return out.sort_values("date").reset_index(drop=True)


def cot_summary(cot, spot):
    last52 = cot.tail(52)

    def blk(col):
        s_, w_ = cot[col].dropna(), last52[col].dropna()
        lo, hi = w_.min(), w_.max()
        return {"net": int(s_.iloc[-1]), "wk_chg": int(s_.iloc[-1] - s_.iloc[-2]),
                "idx": round(float((s_.iloc[-1] - lo) / (hi - lo) * 100)) if hi > lo else 50,
                "pct52": round(float((w_ <= s_.iloc[-1]).mean() * 100)),
                "notional": float(s_.iloc[-1] * 100 * spot)}
    return {"as_of": cot["date"].iloc[-1].strftime("%d/%m"), "swap": blk("swap_net"), "mm": blk("mm_net"),
            "series": [[d.strftime("%Y-%m-%d"), float(a), float(b)]
                       for d, a, b in zip(last52["date"], last52["swap_net"], last52["mm_net"])]}


def smart_score(cot_sum, summ):
    """APPROXIMATE reconstruction: 0.7 x swap-dealer index + 0.5 x options premium skew.
    Weights were fitted to two screenshots of the original dashboard; not validated."""
    sw = (cot_sum["swap"]["idx"] - 50) / 50
    c, p = summ["call_premium_usd_approx"], summ["put_premium_usd_approx"]
    skew = (c - p) / (c + p) if (c + p) > 0 else 0.0
    v = max(-1.0, min(1.0, 0.7 * sw + 0.5 * skew))
    return {"value": round(v, 2), "label": "Bullish" if v >= 0.35 else "Bearish" if v <= -0.35 else "Neutral",
            "swap_component": round(0.7 * sw, 2), "options_component": round(0.5 * skew, 2)}


def fetch_px():
    import yfinance as yf
    h = yf.Ticker("GC=F").history(period="6mo", interval="1d", timeout=15)
    if h.empty:
        raise ValueError("no GC=F history")
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h


def funds_zones(cot, px, shift, spot, lookback, top, min_chg, half=0.005, center_mode="mid"):
    """ESTIMATED zones: the biggest weekly Managed Money net changes in the last `lookback` weeks.
    Zone = that COT week's mid-range (default) or volume-weighted price, shifted to spot, +/- 0.5%."""
    c = cot.copy()
    c["chg"] = c["mm_net"].diff()
    c = c.iloc[1:].tail(lookback)
    c = c[c["chg"].abs() >= min_chg]
    c = c.loc[c["chg"].abs().sort_values(ascending=False).index].head(top)
    rows = []
    for i, r in c.iterrows():
        d, prev_d = r["date"], cot["date"].iloc[i - 1]
        w = px[(px.index > prev_d) & (px.index <= d)]
        if w.empty:
            continue
        if center_mode == "mid":
            center = float((w["High"].max() + w["Low"].min()) / 2 + shift)
        else:
            typ = ((w["High"] + w["Low"] + w["Close"]) / 3 + shift).to_numpy()
            vol = w["Volume"].to_numpy()
            center = float(np.average(typ, weights=vol)) if vol.sum() > 0 else float(typ.mean())
        rows.append({"tag": "Funds BUY" if r["chg"] > 0 else "Funds SELL",
                     "zone_lo": round(center * (1 - half), 2), "zone_hi": round(center * (1 + half), 2),
                     "notional_usd": abs(r["chg"]) * 100 * center, "cot_date": d.strftime("%d/%m/%y"), "chg": int(r["chg"])})
    return rows


# ------------------------------------------------------------ PAXG (24/7 gold token)
BINANCE = ["https://api.binance.com", "https://data-api.binance.vision"]


def fetch_klines(symbol, start_ms, interval="15m"):
    err = None
    for base in BINANCE:
        try:
            rows, t = [], start_ms
            while True:
                r = requests.get(f"{base}/api/v3/klines", timeout=20,
                                 params={"symbol": symbol, "interval": interval, "startTime": t, "limit": 1000})
                r.raise_for_status()
                b = r.json()
                if not b:
                    break
                rows += b
                t = b[-1][6] + 1
                if len(b) < 1000:
                    break
            if rows:
                k = pd.DataFrame(rows).iloc[:, :11]
                k.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                             "qvol", "trades", "taker_buy", "taker_buy_q"]
                for c in ("open", "high", "low", "close", "volume", "taker_buy"):
                    k[c] = pd.to_numeric(k[c])
                k["t"] = pd.to_datetime(k["open_time"], unit="ms", utc=True)
                return k
        except Exception as e:
            err = e
    raise RuntimeError(f"Binance klines failed: {err}")


def vol_profile(sub, bin_=1.0, va=0.70):
    """Volume profile from bars: POC (busiest price) and the value area (70% of volume) high / low."""
    if sub is None or len(sub) < 5 or sub["volume"].sum() <= 0:
        return None
    edges = np.arange(np.floor(sub["low"].min() / bin_) * bin_, np.ceil(sub["high"].max() / bin_) * bin_ + bin_, bin_)
    vol = np.zeros(len(edges) - 1)
    for h, l, v in zip(sub["high"], sub["low"], sub["volume"]):
        i0 = max(0, int((l - edges[0]) // bin_))
        i1 = min(len(vol) - 1, int((h - edges[0]) // bin_))
        vol[i0:i1 + 1] += v / (i1 - i0 + 1)
    poc = int(vol.argmax())
    tot, acc, lo_i, hi_i = vol.sum(), vol[poc], poc, poc
    while acc < va * tot and (lo_i > 0 or hi_i < len(vol) - 1):
        up = vol[hi_i + 1] if hi_i < len(vol) - 1 else -1.0
        dn = vol[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            acc += up
        else:
            lo_i -= 1
            acc += dn
    return {"poc": float(edges[poc] + bin_ / 2), "vah": float(edges[hi_i] + bin_), "val": float(edges[lo_i])}


def paxg_stats(symbol, spot, now, q_utc=None):
    """Anchored VWAPs (typical price x volume) from a 24/7 gold token, shifted to spot by current premium."""
    ms = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo else pd.Timestamp(now, tz="UTC")
    month_start = ms.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    week_start = (ms - pd.Timedelta(days=ms.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    k = fetch_klines(symbol, int(min(month_start, week_start).timestamp() * 1000))
    k["typ"] = (k["high"] + k["low"] + k["close"]) / 3
    last = float(k["close"].iloc[-1])
    prem = last - spot

    def vw(sub):
        v = sub["volume"].sum()
        return float((sub["typ"] * sub["volume"]).sum() / v) if v > 0 else None

    wk, mo = k[k["t"] >= week_start], k[k["t"] >= month_start]
    vwk, vmo = vw(wk), vw(mo)
    paq = None
    if q_utc is not None:   # PAXG price at the moment of GLD's last trade
        qts = pd.Timestamp(q_utc, tz="UTC")
        b = k[k["t"] <= qts]
        if len(b) and (qts - b["t"].iloc[-1]) < pd.Timedelta(minutes=20):
            paq = float(b["close"].iloc[-1])
    buy = float(wk["taker_buy"].sum())
    tot = float(wk["volume"].sum())
    prof = {}
    for nm, sub in (("Wk", wk), ("Mo", mo)):          # volume profile levels, shifted to spot terms
        try:
            vp = vol_profile(sub)
        except Exception:
            vp = None
        if vp:
            for key, lab in (("poc", "POC"), ("vah", "VAH"), ("val", "VAL")):
                prof[f"{nm} {lab}"] = round(vp[key] - prem, 2)
    return {"symbol": symbol, "profile": prof, "last": last, "premium": prem, "premium_pct": prem / spot * 100,
            "vwap_week_token": vwk, "vwap_month_token": vmo,
            "vwap_week_spot": round(vwk - prem, 2) if vwk else None,
            "vwap_month_spot": round(vmo - prem, 2) if vmo else None,
            "week_volume_oz": tot, "week_buy_pct": round(buy / tot * 100, 1) if tot else None,
            "week_delta_oz": buy - (tot - buy), "price_at_quote": paq, "last_bar_utc": k["t"].iloc[-1].strftime("%d/%m %H:%M")}


def paxg_daily(symbol, days=400):
    start = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
    k = fetch_klines(symbol, start, interval="1d")
    out = k[["high", "low", "close", "volume"]].copy()
    out.columns = ["High", "Low", "Close", "Volume"]
    out.index = k["t"].dt.tz_localize(None).dt.normalize()
    return out


# -------------------------------------------------------------------- table
def place(lo, hi, spot):
    if lo <= spot <= hi:
        return 0.0, "at"
    return (lo - spot, "above") if lo > spot else (spot - hi, "below")


def build_table(df, gld_px, ratio, spot, specials, funds, rows, rng, new_min, wall_rng=250.0):
    ks = np.sort(df["strike"].unique())
    step = float(np.min(np.diff(ks))) if len(ks) > 1 else 1.0
    c = df[df["is_call"]].groupby("strike").agg(c_oi=("open_interest", "sum"), c_vol=("volume", "sum"))
    p = df[~df["is_call"]].groupby("strike").agg(p_oi=("open_interest", "sum"), p_vol=("volume", "sum"))
    t = c.join(p, how="outer").fillna(0.0)
    has_d = "oi_delta" in df.columns
    t["d_c"] = df[df["is_call"]].groupby("strike")["oi_delta"].sum() if has_d else 0.0
    t["d_p"] = df[~df["is_call"]].groupby("strike")["oi_delta"].sum() if has_d else 0.0
    t[["d_c", "d_p"]] = t[["d_c", "d_p"]].fillna(0.0)
    t["oi"], t["vol"] = t["c_oi"] + t["p_oi"], t["c_vol"] + t["p_vol"]
    t["expiry"] = df.sort_values("open_interest").groupby("strike")["expiry"].last()
    t = t[t["oi"] > 0]
    med, mx = float(t["oi"].median()) or 1.0, float(t["oi"].max()) or 1.0

    xs = t.index.to_numpy() * ratio
    inr = np.abs(xs - spot) <= wall_rng
    below, above = t[(xs < spot) & inr], t[(xs > spot) & inr]
    pw = below["p_oi"].idxmax() if len(below) and below["p_oi"].max() > 0 else None
    cw = above["c_oi"].idxmax() if len(above) and above["c_oi"].max() > 0 else None
    walls = {"put_wall_xauusd": round(pw * ratio, 2) if pw is not None else None,
             "put_wall_oi": int(below.loc[pw, "p_oi"]) if pw is not None else None,
             "call_wall_xauusd": round(cw * ratio, 2) if cw is not None else None,
             "call_wall_oi": int(above.loc[cw, "c_oi"]) if cw is not None else None}

    items, chart = [], []
    for k, r in t.iterrows():
        lo, hi = (k - step / 2) * ratio, (k + step / 2) * ratio
        dist, pos = place(lo, hi, spot)
        if dist > rng:
            continue
        call_dom = r["c_oi"] >= r["p_oi"]
        if pos == "at":
            sig = "AT PRICE"
        elif pos == "above":
            sig = "RESISTANCE" if call_dom else "ITM PUTS"
        else:
            sig = "SUPPORT" if not call_dom else "ITM CALLS"
        tags = ["Call OI" if call_dom else "Put OI"]
        if k == pw:
            tags.append("Put Wall")
        if k == cw:
            tags.append("Call Wall")
        if has_d and r["d_c"] >= new_min:
            tags.append("New CALL")
        if has_d and r["d_p"] >= new_min:
            tags.append("New PUT")
        items.append({
            "kind": "strike", "tags": tags, "pos": pos, "signal": sig, "gld_strike": float(k),
            "zone_lo": round(lo, 2), "zone_hi": round(hi, 2), "dist": round(dist, 2),
            "dist_pct": round(dist / spot * 100, 2), "call_oi": int(r["c_oi"]), "put_oi": int(r["p_oi"]),
            "oi": int(r["oi"]), "d_call": int(r["d_c"]) if has_d else None, "d_put": int(r["d_p"]) if has_d else None,
            "volume": int(r["vol"]), "notional_usd": float(r["oi"] * CONTRACT_MULT * gld_px),
            "expiry": r["expiry"].strftime("%d/%m"), "strength_x": round(r["oi"] / med, 1),
            "bars": int(min(5, max(1, np.ceil(5 * r["oi"] / mx)))), "oi_frac": float(r["oi"] / mx)})
        chart.append({"xau": round(k * ratio, 2), "c_oi": int(r["c_oi"]), "p_oi": int(r["p_oi"]), "pos": pos,
                      "strength_x": round(r["oi"] / med, 1)})

    for n_, i_ in enumerate(sorted(range(len(items)), key=lambda i: -items[i]["oi"]), 1):   # rank 1 = most contracts
        items[i_]["rank"] = chart[i_]["rank"] = n_
    shown = (sorted([i for i in items if i["pos"] == "above"], key=lambda i: -i["oi"])[:rows]
             + [i for i in items if i["pos"] == "at"]
             + sorted([i for i in items if i["pos"] == "below"], key=lambda i: -i["oi"])[:rows])

    sigmap = {"Gamma flip": "REGIME", "Max pain": "MAGNET?", "VWAP W": "VWAP", "VWAP M": "VWAP"}
    sigmap.update({f"{a_} {b_}": "VPROF" for a_ in ("Wk", "Mo") for b_ in ("POC", "VAH", "VAL")})
    for name, px_ in specials:                      # merge into the strike zone that contains it
        if px_ is None or abs(px_ - spot) > rng:
            continue
        host = next((r for r in shown if r["kind"] == "strike" and r["zone_lo"] <= px_ <= r["zone_hi"]), None)
        if host:
            host["tags"].append(name)
        else:
            shown.append({"kind": "special", "tags": [name], "signal": sigmap[name], "zone_lo": round(px_, 2),
                          "zone_hi": round(px_, 2), "dist": round(abs(px_ - spot), 2), "pos": "above" if px_ > spot else "below",
                          "dist_pct": round(abs(px_ - spot) / spot * 100, 2)})

    for r in shown:   # heuristic action tier: strong strikes or walls
        if r["kind"] == "strike" and r["signal"] in ("RESISTANCE", "SUPPORT") and (
                r["strength_x"] >= 3.0 or "Call Wall" in r["tags"] or "Put Wall" in r["tags"]):
            r["signal"] = "SELL ZONE" if r["signal"] == "RESISTANCE" else "BUY ZONE"
    fmed = float(np.median([f["notional_usd"] for f in funds])) if funds else 1.0
    fmax = max([f["notional_usd"] for f in funds], default=1.0)
    skipped = 0
    for f in funds:
        dist, pos = place(f["zone_lo"], f["zone_hi"], spot)
        if dist > rng:
            skipped += 1
            continue
        buy = f["tag"] == "Funds BUY"
        sig = ("AT PRICE" if pos == "at" else "SUPPORT" if buy and pos == "below" else
               "RESISTANCE" if (not buy) and pos == "above" else "UNDERWATER")
        shown.append({"kind": "funds", "tags": [f["tag"]], "pos": pos, "signal": sig, "zone_lo": f["zone_lo"],
                      "zone_hi": f["zone_hi"], "dist": round(dist, 2), "dist_pct": round(dist / spot * 100, 2),
                      "notional_usd": f["notional_usd"], "expiry": "COT " + f["cot_date"], "chg": f["chg"],
                      "strength_x": round(f["notional_usd"] / fmed, 1),
                      "bars": int(min(5, max(1, np.ceil(5 * f["notional_usd"] / fmax))))})
    shown.sort(key=lambda i: -(i["zone_lo"] + i["zone_hi"]))
    chart.sort(key=lambda i: -i["xau"])
    return shown, chart, walls, skipped


def summarise(df):
    c, p = df[df["is_call"]], df[~df["is_call"]]
    c_oi, p_oi, c_vol, p_vol = c["open_interest"].sum(), p["open_interest"].sum(), c["volume"].sum(), p["volume"].sum()
    return {"call_oi": int(c_oi), "put_oi": int(p_oi), "pc_ratio_oi": round(p_oi / c_oi, 3) if c_oi else None,
            "call_volume": int(c_vol), "put_volume": int(p_vol),
            "pc_ratio_volume": round(p_vol / c_vol, 3) if c_vol else None,
            "call_premium_usd_approx": round(float((c["volume"] * c["mid"] * CONTRACT_MULT).sum())),
            "put_premium_usd_approx": round(float((p["volume"] * p["mid"] * CONTRACT_MULT).sum()))}


# --------------------------------------------------------------------- html
def build_watchlist(out, min_x, scalp):
    """Nearest strong strike zones above/below price, the zone price is in, and the marker prices."""
    spot = out["spot_xauusd"]
    strikes = [r for r in out["levels"] if r["kind"] == "strike"]
    strong = [r for r in strikes if r["strength_x"] >= min_x]
    above = sorted([r for r in strong if r["pos"] == "above"], key=lambda r: r["dist"])
    below = sorted([r for r in strong if r["pos"] == "below"], key=lambda r: r["dist"])
    here = next((r for r in strikes if r["pos"] == "at"), None)
    marks = []
    for name, px_ in (("Gamma flip", out["gamma_flip_xauusd"]), ("Max pain", out["max_pain_xauusd"]),
                      ("VWAP W", out["vwap_week"]), ("VWAP M", out["vwap_month"])):
        if px_:
            marks.append({"name": name, "px": round(float(px_), 2), "dist": round(float(px_) - spot, 2)})
    marks.sort(key=lambda m: abs(m["dist"]))
    return {"above": above[:2], "below": below[:2], "here": here, "marks": marks, "min_x": min_x, "scalp": scalp}


def watch_html(w, spot):
    own = lambda t: not (t.startswith("VWAP") or t in ("Gamma flip", "Max pain"))

    def stacked(r):
        near = [plain(m["name"]) for m in w["marks"] if r["zone_lo"] - 10 <= m["px"] <= r["zone_hi"] + 10]
        return f'<div class="wline">Lined up with: <b>{html.escape(", ".join(near))}</b></div>' if near else ""

    def card(title, items, side):
        if not items:
            return (f'<div class="wcard"><h3>{title}</h3><div class="meta">No level of x{w["min_x"]:g} or more '
                    f'in the table range.</div></div>')
        r = items[0]
        cls, sym = ("up", "▲") if side == "above" else ("dn", "▼")
        far = r["dist"] > w["scalp"]
        note = (f'<span class="wfar">Far (more than ${w["scalp"]:g}): a target or caution zone, not a scalp entry.</span>'
                if far else f'<span class="wnear">Within ${w["scalp"]:g}: close enough to matter for a scalp.</span>')
        tags = "".join(chip(t) for t in r["tags"] if own(t))
        nxt = ""
        if len(items) > 1:
            n = items[1]
            nxt = (f'<div class="wnote">Then: {n["zone_lo"]:,.2f} – {n["zone_hi"]:,.2f} · {sym} {n["dist"]:.2f} · '
                   f'x{n["strength_x"]:.1f}</div>')
        return (f'<div class="wcard"><h3>{title}</h3><div><span class="wzone">{r["zone_lo"]:,.2f} – {r["zone_hi"]:,.2f}</span>'
                f'<span class="wdist {cls}">{sym} ${r["dist"]:.2f}</span></div>'
                f'<div class="wline">Rank <b>#{r["rank"]}</b> · <b>{r["strength_x"]:.1f}×</b> a typical level · {r["oi"]:,} contracts open · exp {r["expiry"]}</div>'
                f'<div class="wline">{tags}</div>{stacked(r)}<div class="wnote">{note}</div>{nxt}</div>')

    h = w["here"]
    if h:
        hx = h["strength_x"]
        hnote = ("Big zone: price often chops or reacts around levels this size." if hx >= 7
                 else "Moderate zone: some structure here, but not a big level." if hx >= w["min_x"]
                 else "Weak zone: not much structure right here.")
        tags = "".join(chip(t) for t in h["tags"] if own(t))
        here = (f'<div class="wcard here"><h3>You are here · {spot:,.2f}</h3><div class="wzone">'
                f'{h["zone_lo"]:,.2f} – {h["zone_hi"]:,.2f}</div>'
                f'<div class="wline">Rank <b>#{h["rank"]}</b> · <b>{h["strength_x"]:.1f}×</b> a typical level · {h["oi"]:,} contracts open · exp {h["expiry"]}</div>'
                f'<div class="wline">{tags}</div>{stacked(h)}<div class="wnote">{hnote}</div></div>')
    else:
        here = (f'<div class="wcard here"><h3>You are here · {spot:,.2f}</h3>'
                f'<div class="meta">Price is between zones in the table.</div></div>')
    marks = " · ".join(f'{plain(m["name"])} <b>{m["px"]:,.2f}</b> ({"▲" if m["dist"] > 0 else "▼"} {abs(m["dist"]):.2f})'
                       for m in w["marks"])
    mk = f'<div class="wmarks">Price markers, nearest first: {marks}</div>' if marks else ""
    return (f'<div class="watch">{card("Nearest strong level above", w["above"], "above")}{here}'
            f'{card("Nearest strong level below", w["below"], "below")}</div>{mk}')


CSS = """
body{background:#08111f;color:#d7e2f2;font:13px system-ui,Segoe UI,sans-serif;margin:14px}
h1{font-size:17px;margin:0 0 4px;border-left:3px solid #35b6ff;padding-left:8px}
.meta{color:#7f95b5;margin:0 0 12px;font-size:12px}
.cards{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.card{background:#0f1c33;border:1px solid #1c2e4d;border-radius:8px;padding:10px 12px;min-width:210px;flex:1}
.card h3{margin:0 0 6px;font-size:11px;color:#7f95b5;font-weight:600;text-transform:uppercase}
.card .meta{margin:2px 0 0}
.card .big{font-size:20px;font-weight:700}.neg,.rn{color:#ff6b6b}.pos,.gp{color:#3ddc84}
.pbar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0}
.pbar span:first-child{background:#3ddc84}.pbar span:last-child{background:#ff6b6b}
.wrap{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
.tbl{flex:1 1 1010px;min-width:0;overflow-x:auto}
table{border-collapse:collapse;width:100%}
th{color:#7f95b5;font-weight:600;font-size:11px;text-align:right;padding:6px 8px;border-bottom:1px solid #1c2e4d}
th:first-child,td:first-child{text-align:left}
td{padding:6px 8px;border-bottom:1px solid #12223d;text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-right:3px}
.t-call{background:#3a1620;color:#ff7a7a}.t-put{background:#10301f;color:#43e08a}
.t-sp{background:#33290f;color:#f5c542}.t-vw{background:#10284a;color:#6fd0ff}
.t-fb{background:#0e3a3a;color:#4fe0d0}.t-fs{background:#3a2a10;color:#ffb347}
.t-newc{border:1px solid #ff7a7a;color:#ff7a7a}.t-newp{border:1px solid #43e08a;color:#43e08a}
.up{color:#ff6b6b}.dn{color:#3ddc84}
.sig{display:inline-block;padding:2px 9px;border-radius:10px;font-size:10.5px;font-weight:700}
.s-res{background:#4a1a22;color:#ff8d8d}.s-sup{background:#12402a;color:#4be58f}
.s-itm,.s-und{background:#1f2937;color:#8da0bb}.s-at{background:#4a3410;color:#ffb347}
.s-reg{background:#33290f;color:#f5c542}.s-vw{background:#10284a;color:#6fd0ff}
.ob{display:inline-block;width:70px;height:6px;background:#12223d;border-radius:3px;vertical-align:middle;margin-right:6px}
.ob b{display:block;height:100%;border-radius:3px;background:#3ddc84}
.str i{display:inline-block;width:3px;margin-right:1px;background:#1f2e4a;vertical-align:bottom}
.str i.on{background:#35b6ff}
tr.at{background:#2a2310;outline:1px solid #ffb347}
tr.spot td{background:#10304f;color:#6fd0ff;font-weight:700;text-align:center;border-bottom:1px solid #35b6ff}
.chart{flex:1;min-width:300px;background:#0f1c33;border:1px solid #1c2e4d;border-radius:8px;padding:8px}
.chart h3{margin:0 0 4px;font-size:11px;color:#7f95b5;text-transform:uppercase}
.foot{color:#6a7f9f;font-size:11px;margin-top:10px;max-width:1000px}
.pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;margin-left:10px;vertical-align:middle}
.p-pos{background:#12402a;color:#4be58f}.p-neg{background:#4a1a22;color:#ff8d8d}.p-neu{background:#1f2937;color:#c3d0e4}
.s-sellz{border:1px solid #ff6b6b;color:#ff8d8d}.s-buyz{border:1px solid #3ddc84;color:#4be58f}
.strip{background:#0b1830;border:1px solid #1c2e4d;border-radius:8px;padding:7px 12px;margin-bottom:12px;color:#8da0bb;font-size:12px}
.strip b{color:#e6eefb}
.gauge{height:6px;border-radius:3px;margin:8px 0 3px;position:relative;background:linear-gradient(90deg,#c0392b,#2c3e5c 50%,#1e9e5a)}
.gauge i{position:absolute;top:-3px;width:10px;height:12px;border-radius:5px;background:#f5c542;margin-left:-5px}
.chg{font-size:11px;font-weight:700;padding:1px 8px;border-radius:9px;background:#14233d}
.card.wide{flex:2;min-width:340px}
.watch{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.wcard{background:#0d1f3a;border:1px solid #2a4a7a;border-radius:8px;padding:10px 14px;min-width:260px;flex:1}
.wcard.here{border-color:#ffb347;background:#2a2310}
.wcard h3{margin:0 0 6px;font-size:11px;color:#8fb4e8;font-weight:700;text-transform:uppercase;letter-spacing:.3px}
.wzone{font-size:19px;font-weight:700;color:#e6eefb}
.wdist{font-size:15px;font-weight:700;margin-left:8px}
.wline{margin-top:5px;color:#c3d0e4}
.wnote{margin-top:6px;font-size:12px;color:#8da0bb}
.wfar{color:#ffb347}.wnear{color:#4be58f}
.wmarks{background:#0b1830;border:1px solid #1c2e4d;border-radius:8px;padding:7px 12px;margin-bottom:12px;color:#8da0bb;font-size:12px}
.wmarks b{color:#e6eefb}
th small{display:block;font-weight:400;color:#5d7396;font-size:9.5px}
td.rk{text-align:center;font-weight:800;color:#f5c542}
td.what{text-align:left}
th:nth-child(2){text-align:left}
tr td{border-left:0}
tr.big{outline:2px solid #ff6b6b;outline-offset:-2px;background:#1a1218}
tr.big.be{outline-color:#3ddc84;background:#0f1d19}
tr.big.nw{outline-color:#ffb347}
tr.mid{outline:1px solid #6b4a55;outline-offset:-1px}
tr.mid.be{outline-color:#3f7a5b}
tr.mid.nw{outline-color:#ffb347}
tr.small{opacity:.5}
tr.small.nw{opacity:1}
tr.mark{outline:1px dashed #5d7396;outline-offset:-1px}
tr.big td:first-child{border-left:6px solid #ff6b6b}
tr.big.be td:first-child{border-left-color:#3ddc84}
tr.big.nw td:first-child{border-left-color:#ffb347}
tr.mid td:first-child{border-left:4px solid #8a4a58}
tr.mid.be td:first-child{border-left-color:#3f7a5b}
tr.big td{padding-top:9px;padding-bottom:9px}
.star{display:inline-block;background:#f5c542;color:#1a1400;font-size:10px;font-weight:800;padding:2px 7px;border-radius:9px;margin-right:5px}
.k-big{color:#fff;background:#b8323a;padding:1px 7px;border-radius:8px;font-size:10.5px}
tr.be .k-big{background:#1e8a52}
.k-mid{color:#e6eefb;background:#34476b;padding:1px 7px;border-radius:8px;font-size:10.5px}
.k-sm{color:#7f95b5;font-size:10.5px}
.plain{background:#0d1f3a;border:1px solid #2a4a7a;border-radius:8px;padding:10px 16px;margin-bottom:12px}
.plain h2{margin:0 0 6px;font-size:13px;color:#8fb4e8;text-transform:uppercase;letter-spacing:.3px}
.plain ul{margin:0;padding-left:18px;line-height:1.6;font-size:13.5px}
.legend{background:#0b1830;border:1px solid #1c2e4d;border-radius:8px;padding:8px 12px;margin-bottom:10px;color:#8da0bb;font-size:12px;line-height:2}
.lg{margin-left:14px;white-space:nowrap}
.sw{display:inline-block;width:26px;height:12px;vertical-align:middle;margin-right:5px;background:#12223d}
.sw.big{border:2px solid #ff6b6b}.sw.mid{border:1px solid #8a4a58}.sw.sm{opacity:.4;border:1px solid #1c2e4d}.sw.mk{border:1px dashed #5d7396}
.rb{border-radius:8px;padding:8px 14px;margin-bottom:10px;font-size:13px;font-weight:600}
.rb-bad{background:#4a1a22;border:1px solid #ff6b6b;color:#ffb0b0}
.rb-warn{background:#33290f;border:1px solid #f5c542;color:#ffe08a}
.rb-ok{background:#10301f;border:1px solid #2f8f5c;color:#7fe6ad;font-weight:500}
.say{margin-top:8px;font-size:12px;color:#c3d0e4;line-height:1.5;border-top:1px solid #1c2e4d;padding-top:6px}
.keys{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}
.key{flex:1;min-width:320px;background:#0d1f3a;border:1px solid #2a4a7a;border-radius:8px;padding:12px 16px}
.key h3{margin:0 0 4px;font-size:12px;color:#8fb4e8;text-transform:uppercase;letter-spacing:.3px}
.kn{font-size:34px;font-weight:800;color:#e6eefb;line-height:1.15}
.kl{font-size:16px;margin:2px 0 6px}
.cexp{font-size:12px;color:#aebcd3;line-height:1.55;margin:4px 0 8px}
.cexp.meta{color:#7f95b5}
.chart{flex:0 0 440px;min-width:440px}
td.what{white-space:normal;min-width:140px}
td{padding:6px 7px}
"""


def money(x):
    x = float(x)
    sg, a = ("-" if x < 0 else ""), abs(x)
    for lim, u in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= lim:
            return f"{sg}${a / lim:,.1f}{u}"
    return f"{sg}${a:,.0f}"


TAG_PLAIN = {"Call OI": "Mostly calls", "Put OI": "Mostly puts", "Put Wall": "Biggest put level",
             "Call Wall": "Biggest call level", "New CALL": "New calls added", "New PUT": "New puts added",
             "VWAP W": "Weekly VWAP", "VWAP M": "Monthly VWAP", "Funds BUY": "Funds bought here",
             "Funds SELL": "Funds sold here",
             "Wk POC": "Week volume peak", "Wk VAH": "Week value-area top", "Wk VAL": "Week value-area bottom",
             "Mo POC": "Month volume peak", "Mo VAH": "Month value-area top", "Mo VAL": "Month value-area bottom"}
SIG_PLAIN = {"RESISTANCE": "Possible ceiling", "SELL ZONE": "Big ceiling", "SUPPORT": "Possible floor",
             "BUY ZONE": "Big floor", "ITM PUTS": "Size only", "ITM CALLS": "Size only", "AT PRICE": "You are here",
             "REGIME": "Speed switch", "MAGNET?": "Possible magnet", "VWAP": "Average price",
             "UNDERWATER": "Funds underwater", "VPROF": "Volume level"}
SIG_TIP = {"RESISTANCE": "Above price, mostly calls: price may stall here on the way up.",
           "SELL ZONE": "Above price, mostly calls and a large pile: price may stall or turn here.",
           "SUPPORT": "Below price, mostly puts: price may bounce here on the way down.",
           "BUY ZONE": "Below price, mostly puts and a large pile: price may bounce here.",
           "ITM PUTS": "Above price but mostly puts, so it is not treated as a ceiling.",
           "ITM CALLS": "Below price but mostly calls, so it is not treated as a floor.",
           "AT PRICE": "Price is inside this zone right now.",
           "REGIME": "Gamma flip: below it moves tend to speed up, above it they tend to calm down.",
           "MAGNET?": "Max pain: the price where option sellers lose least. A weak pull, mainly near expiry.",
           "VWAP": "Average traded price for the week or month (from the PAXG gold token).",
           "VPROF": "Volume profile (PAXG gold token): the peak is the price with the most trading; the value area holds about 70% of the volume. Price often reacts at these.",
           "UNDERWATER": "Funds are probably losing on a position opened around here."}


def plain(t):
    return TAG_PLAIN.get(t, t)


def chip(t):
    cls = ("t-newc" if t == "New CALL" else "t-newp" if t == "New PUT" else "t-fb" if t == "Funds BUY" else
           "t-fs" if t == "Funds SELL" else "t-vw" if t.startswith(("VWAP", "Wk ", "Mo ")) else
           "t-call" if t.startswith("Call") else "t-put" if t.startswith("Put") else "t-sp")
    return f'<span class="tag {cls}">{html.escape(plain(t))}</span>'


def bars_html(n):
    return "".join(f'<i class="{"on" if i < n else ""}" style="height:{4 + i * 3}px"></i>' for i in range(5))


def tier(r, min_x):
    if r["kind"] != "strike":
        return "mark"
    x = r["strength_x"]
    return "big" if x >= 7 else "mid" if x >= min_x else "small"


def size_label(x, min_x):
    return ("BIG", "k-big") if x >= 7 else ("MEDIUM", "k-mid") if x >= min_x else ("small", "k-sm")


HEADS = {"rank": "Rank<small>by size</small>", "what": "What it is", "zone": "Price area<small>XAUUSD</small>",
         "dist": "How far<small>$ from price</small>", "pct": "%", "big": "How big<small>vs a typical level</small>",
         "oi": "Contracts open<small>OI</small>", "cp": "Calls / Puts<small>contracts open</small>",
         "delta": "Added overnight<small>OI Δ · calls / puts</small>", "exp": "Expires",
         "vol": "Traded today<small>volume</small>", "notional": "Size in gold<small>notional $</small>",
         "sig": "What it may do<small>signal</small>"}


def row_html(r, cols, min_x, star=False):
    sig = r["signal"]
    scls = {"RESISTANCE": "s-res", "SUPPORT": "s-sup", "AT PRICE": "s-at", "REGIME": "s-reg", "MAGNET?": "s-reg",
            "VWAP": "s-vw", "VPROF": "s-vw", "UNDERWATER": "s-und", "SELL ZONE": "s-sellz", "BUY ZONE": "s-buyz"}.get(sig, "s-itm")
    side = "nw" if r["pos"] == "at" else "ab" if r["pos"] == "above" else "be"
    trc = f"{tier(r, min_x)} {side}"
    c = {}
    c["rank"] = f'<td class="rk">{"#" + str(r["rank"]) if "rank" in r else "–"}</td>'
    badge = '<span class="star">★ WATCH</span>' if star else ""
    c["what"] = f'<td class="what">{badge}{"".join(chip(t) for t in r["tags"])}</td>'
    c["zone"] = (f'<td>{r["zone_lo"]:,.2f}</td>' if r["kind"] == "special"
                 else f'<td>{r["zone_lo"]:,.2f} – {r["zone_hi"]:,.2f}</td>')
    if r["pos"] == "at":
        c["dist"] = "<td>0.00</td>"
    else:
        arrow, sym = ("up", "▲") if r["pos"] == "above" else ("dn", "▼")
        c["dist"] = f'<td class="{arrow}">{sym} {r["dist"]:.2f}</td>'
    c["pct"] = f'<td>{r["dist_pct"]:.2f}%</td>'
    c["sig"] = (f'<td><span class="sig {scls}" title="{html.escape(SIG_TIP.get(sig, ""))}">'
                f'{html.escape(SIG_PLAIN.get(sig, sig))}</span></td>')
    if r["kind"] != "special":
        lab, kc = size_label(r["strength_x"], min_x)
        c["big"] = (f'<td class="str"><b class="{kc}">{lab}</b> {bars_html(r["bars"])} '
                    f'<span class="meta">x{r["strength_x"]:.1f}</span></td>')
    if r["kind"] == "strike":
        c["oi"] = f'<td><span class="ob"><b style="width:{r["oi_frac"] * 100:.0f}%"></b></span>{r["oi"]:,}</td>'
        c["cp"] = f'<td><span class="rn">{r["call_oi"]:,}</span> / <span class="gp">{r["put_oi"]:,}</span></td>'
        if r["d_call"] is None:
            c["delta"] = "<td>–</td>"
        else:
            f = lambda v: f'<span class="{"gp" if v > 0 else "rn" if v < 0 else ""}">{v:+,}</span>'
            c["delta"] = f'<td>{f(r["d_call"])} / {f(r["d_put"])}</td>'
        c["exp"] = f'<td>{r["expiry"]}</td>'
        c["vol"] = f'<td>{r["volume"]:,}</td>'
        c["notional"] = f'<td>{money(r["notional_usd"])}</td>'
    i0, isg = cols.index("big"), cols.index("sig")
    if r["kind"] == "strike":
        cells = "".join(c[k] for k in cols)
    elif r["kind"] == "special":
        cells = ("".join(c[k] for k in cols[:i0]) + f'<td colspan="{isg - i0}" class="meta"></td>' + c["sig"])
    else:
        buy = "BUY" in r["tags"][0]
        txt = (f'estimate: about {abs(r["chg"]):,} contracts {"bought" if buy else "sold"} that week · '
               f'≈ {money(r["notional_usd"])} · {r["expiry"]}')
        cells = ("".join(c[k] for k in cols[:i0 + 1]) +
                 f'<td colspan="{isg - i0 - 1}" class="meta" style="text-align:left">{txt}</td>' + c["sig"])
    return f'<tr class="{trc}">{cells}</tr>'


def render_chart(chart, spot, marks):
    """OI by strike: puts (green) to the left, calls (red) to the right, ranked, with price markers on the right."""
    if not chart:
        return ""
    mx = max(max(c["c_oi"], c["p_oi"]) for c in chart) or 1
    rh, t0, mid, bw, W = 20, 26, 215, 100, 440
    H = t0 + rh * len(chart) + 8
    prices = [c["xau"] for c in chart]

    def ypos(px_):
        if px_ >= prices[0]:
            return t0 + rh / 2
        if px_ <= prices[-1]:
            return t0 + (len(prices) - 0.5) * rh
        for i in range(len(prices) - 1):
            if prices[i] >= px_ >= prices[i + 1]:
                return t0 + (i + (prices[i] - px_) / ((prices[i] - prices[i + 1]) or 1) + 0.5) * rh
        return t0

    fk = lambda v: f"{v / 1000:.1f}K" if v >= 1000 else f"{v}"
    s = f'<svg viewBox="0 0 {W} {H}" width="100%" font-size="10" font-family="system-ui">'
    s += (f'<text x="{mid - 6}" y="14" text-anchor="end" fill="#4be58f" font-weight="700">◀ PUTS (green)</text>'
          f'<text x="{mid + 6}" y="14" fill="#ff8d8d" font-weight="700">CALLS (red) ▶</text>'
          f'<text x="0" y="14" fill="#7f95b5">Price</text><text x="40" y="14" fill="#7f95b5">Rank</text>')
    for i, c in enumerate(chart):
        y = t0 + i * rh
        tint = {"ab": ("#e05a5a", ".08"), "be": ("#2fbf71", ".08"), "at": ("#ffb347", ".18")}[
            "at" if c["pos"] == "at" else "ab" if c["pos"] == "above" else "be"]
        bright = c["rank"] <= 5
        op = 1 if bright else .4
        pw, cw = c["p_oi"] / mx * bw, c["c_oi"] / mx * bw
        s += (f'<g><title>{c["xau"]:,.0f} zone · rank #{c["rank"]} · {c["c_oi"]:,} call contracts, {c["p_oi"]:,} put '
              f'contracts · {c["strength_x"]:.1f}x a typical level</title>'
              f'<rect x="0" y="{y}" width="340" height="{rh}" fill="{tint[0]}" opacity="{tint[1]}"/>'
              f'<text x="0" y="{y + 14}" fill="{"#e6eefb" if bright else "#8da0bb"}" font-weight="{700 if bright else 400}">'
              f'{c["xau"]:,.0f}</text>')
        if bright:
            s += (f'<rect x="38" y="{y + 3}" width="24" height="14" rx="7" fill="#f5c542"/>'
                  f'<text x="50" y="{y + 13.5}" text-anchor="middle" fill="#1a1400" font-weight="800" font-size="9">#{c["rank"]}</text>')
        s += (f'<rect x="{mid - pw:.1f}" y="{y + 3}" width="{pw:.1f}" height="{rh - 6}" fill="#2fbf71" opacity="{op}"/>'
              f'<rect x="{mid:.1f}" y="{y + 3}" width="{cw:.1f}" height="{rh - 6}" fill="#e05a5a" opacity="{op}"/>')
        if bright:
            if c["p_oi"] >= c["c_oi"]:
                s += f'<text x="{mid - pw - 3:.1f}" y="{y + 14}" text-anchor="end" fill="#4be58f" font-weight="700">{fk(c["p_oi"])}</text>'
            else:
                s += f'<text x="{mid + cw + 3:.1f}" y="{y + 14}" fill="#ff8d8d" font-weight="700">{fk(c["c_oi"])}</text>'
        s += "</g>"
    s += f'<line x1="{mid}" x2="{mid}" y1="{t0}" y2="{H - 8}" stroke="#2a3b5a"/>'
    pts = sorted([(ypos(px_), nm, px_, col, bg) for nm, px_, col, bg in marks
                  if px_ and prices[-1] - 12 <= px_ <= prices[0] + 12], key=lambda m: m[0])
    last = -99
    for ty, nm, px_, col, bg in pts:
        py = max(ty, last + 15)
        last = py
        dash = "" if nm == "NOW" else ' stroke-dasharray="3 3"'
        s += (f'<line x1="66" x2="340" y1="{ty:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="{1.6 if nm == "NOW" else 1}"{dash} opacity=".9"/>'
              f'<polyline points="340,{ty:.1f} 344,{py:.1f}" stroke="{col}" fill="none"/>'
              f'<rect x="346" y="{py - 7:.1f}" width="92" height="14" rx="7" fill="{bg}" stroke="{col}"/>'
              f'<text x="392" y="{py + 3:.1f}" text-anchor="middle" fill="{col}" font-size="9" font-weight="700">{nm} {px_:,.0f}</text>')
    return s + "</svg>"


def chart_marks(out):
    sp = out["spot_xauusd"]
    return [("NOW", sp, "#35b6ff", "#10284a"), ("γ FLIP", out["gamma_flip_xauusd"], "#b79bff", "#2a1f4a"),
            ("MAX PAIN", out["max_pain_xauusd"], "#f5c542", "#33290f"),
            ("WEEK VWAP", out["vwap_week"], "#6fd0ff", "#0e2a45"), ("MONTH VWAP", out["vwap_month"], "#4fe0d0", "#0e3a3a")]


def chart_html(out):
    ch, sp = out["chart"], out["spot_xauusd"]
    if not ch:
        return ""
    above = [c for c in ch if c["pos"] == "above"]
    below = [c for c in ch if c["pos"] == "below"]
    ba = max(above, key=lambda c: c["c_oi"] + c["p_oi"], default=None)
    bb = max(below, key=lambda c: c["c_oi"] + c["p_oi"], default=None)
    now = []
    for lab, c, sym in (("above", ba, "▲"), ("below", bb, "▼")):
        if c:
            now.append(f'Biggest pile {lab} price: <b>{c["xau"]:,.0f}</b> (rank #{c["rank"]}, {sym} about ${abs(c["xau"] - sp):.0f} away)')
    return (f'<div class="chart"><h3>Where the option contracts are piled up</h3>'
            f'<div class="cexp">Each row is a price zone, highest price at the top. <b>Longer bar = more contracts = a bigger level.</b> '
            f'<span class="gp">Green bars (left) = puts</span>, <span class="rn">red bars (right) = calls</span>. '
            f'The <b>5 biggest levels are bright with a gold #rank</b>; the rest are faded. Rows tinted red are above price (possible ceilings), '
            f'green are below (possible floors). The blue line is where gold is now.</div>'
            f'{render_chart(ch, sp, chart_marks(out))}'
            f'<div class="cexp">{" · ".join(now)}</div>'
            f'<div class="cexp meta">Note: this shows how many contracts exist, not whether buyers or sellers are winning. '
            f'It is a map of places where price tends to react, not a prediction of direction.</div></div>')


def plain_summary(out):
    w = out.get("watch")
    if not w:
        return ""
    spot, n = out["spot_xauusd"], len(out["chart"])
    strikes = [r for r in out["levels"] if r["kind"] == "strike"]
    width = (strikes[0]["zone_hi"] - strikes[0]["zone_lo"]) if strikes else 11

    def lvl(r, word):
        near = [plain(m["name"]) for m in w["marks"] if r["zone_lo"] - 10 <= m["px"] <= r["zone_hi"] + 10]
        s = (f'<b>{r["zone_lo"]:,.2f} – {r["zone_hi"]:,.2f}</b>, ${r["dist"]:.2f} {word} price '
             f'(rank #{r["rank"]} of {n}, {r["strength_x"]:.1f}× a typical level)')
        return s + (f', lined up with {", ".join(near)}' if near else "")

    li, h = [], w["here"]
    if h:
        x = h["strength_x"]
        sz = ("a <b>big</b> level, so price often reacts or chops around here" if x >= 7 else
              "a <b>medium</b> level" if x >= w["min_x"] else "a <b>weak</b> level, not much happening right at this price")
        li.append(f'Gold is at <b>{spot:,.2f}</b>, inside {sz}.')
    else:
        li.append(f'Gold is at <b>{spot:,.2f}</b>, between zones.')
    if w["above"]:
        li.append(f'<b class="up">Closest strong level ABOVE</b> ({"possible ceiling" if w["above"][0]["signal"] in ("SELL ZONE", "RESISTANCE") else "mostly puts, so more of a pivot area than a clean ceiling"}): {lvl(w["above"][0], "above")}. '
                  f'Price may slow down, turn back, or push straight through. Watch how it reacts.')
    else:
        li.append('<b class="up">Above:</b> no strong level in the table range, nothing special overhead.')
    if w["below"]:
        li.append(f'<b class="dn">Closest strong level BELOW</b> ({"possible floor" if w["below"][0]["signal"] in ("BUY ZONE", "SUPPORT") else "mostly calls, so not a clean floor"}): {lvl(w["below"][0], "below")}. '
                  f'Price may bounce there, or fall through.')
    else:
        li.append('<b class="dn">Below:</b> no strong level in the table range, nothing special underneath.')
    flip_ = out["gamma_flip_xauusd"]
    if abs(out["net_gex_usd_per_1pct"]) < 25e6 or (flip_ and abs(flip_ - spot) <= 8):
        li.append('<b>Speed:</b> price is at or very near the gamma flip, or total gamma is close to zero, so there is <b>no clear speed edge</b> right now. '
                  'Expect normal behaviour and see how price reacts. (Estimate based on an assumption.)')
    else:
        li.append('<b>Speed:</b> ' + (
        'market makers are positioned so their hedging likely <b>speeds moves up</b> right now (negative gamma): expect faster, bigger swings and give trades more room.'
        if out["net_gex_usd_per_1pct"] < 0 else
        'market makers are positioned so their hedging likely <b>calms moves down</b> right now (positive gamma): expect more chop and mean-reversion.')
        + ' (This is an estimate based on an assumption.)')
    li.append(f'<b>Reminder:</b> each zone is about ${width:.0f} wide, so these are areas to pay attention to, not exact prices. '
              f'Wait for your own price-action signal (rejection candle, break and retest) before trading.')
    return '<div class="plain"><h2>In plain words</h2><ul>' + "".join(f"<li>{x}</li>" for x in li) + "</ul></div>"


def legend_html():
    return ('<div class="legend"><b>How to read the table:</b> '
            '<span class="lg"><i class="sw big"></i>thick bright border = BIG level (7× typical or more)</span>'
            '<span class="lg"><i class="sw mid"></i>thin border = MEDIUM (3–7×)</span>'
            '<span class="lg"><i class="sw sm"></i>faded = small, mostly ignore</span>'
            '<span class="lg"><i class="sw mk"></i>dashed = marker or estimate (VWAP, flip, funds)</span>'
            '<span class="lg"><span class="star">★ WATCH</span> nearest strong level on each side</span>'
            '<span class="lg"><b class="up">red side bar</b> = above price (ceiling) · <b class="dn">green side bar</b> = below (floor)</span>'
            '<span class="lg"><b>#1</b> = biggest pile of contracts · <b>x9.5</b> = 9.5× a typical level</span></div>')


def render_cot_chart(series):
    if not series or len(series) < 3:
        return ""
    W, H, L, R, T, B = 330, 125, 36, 40, 8, 16
    sw = np.array([r[1] for r in series])
    mm = np.array([r[2] for r in series])
    lo, hi = min(sw.min(), mm.min(), 0), max(sw.max(), mm.max(), 0)
    X = lambda i: L + i / (len(series) - 1) * (W - L - R)
    Y = lambda v: T + (hi - v) / ((hi - lo) or 1) * (H - T - B)
    path = lambda arr: "M" + " L".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(arr))
    s = f'<svg viewBox="0 0 {W} {H}" width="100%" font-size="9" font-family="system-ui">'
    s += f'<line x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="#2a3b5a" stroke-dasharray="3 3"/>'
    for v in (hi, 0, lo):
        s += f'<text x="0" y="{Y(v) + 3:.1f}" fill="#7f95b5">{v / 1000:+.0f}K</text>'
    s += f'<path d="{path(sw)}" fill="none" stroke="#4aa3ff" stroke-width="1.6"/>'
    s += f'<path d="{path(mm)}" fill="none" stroke="#f5a623" stroke-width="1.6"/>'
    for arr, col in ((sw, "#4aa3ff"), (mm, "#f5a623")):
        s += f'<text x="{W - R + 4}" y="{Y(arr[-1]) + 3:.1f}" fill="{col}" font-weight="700">{arr[-1] / 1000:+.0f}K</text>'
    s += f'<text x="{L}" y="{H - 2}" fill="#7f95b5">{series[0][0][:7]}</text>'
    s += f'<text x="{W - R}" y="{H - 2}" text-anchor="end" fill="#7f95b5">{series[-1][0][:7]}</text></svg>'
    return s


def ratio_banner(out):
    w = out.get("ratio_warn")
    if not w:
        return ""
    k, t = w
    return f'<div class="rb rb-{k}">{"✔" if k == "ok" else "⚠"} {html.escape(t)}</div>'


def say(t):
    return f'<div class="say">{t}</div>'


def paxg_say(px):
    b = px["week_buy_pct"] or 50
    tilt = "slightly more aggressive selling than buying" if b < 48 else \
        "slightly more aggressive buying than selling" if b > 52 else "balanced buying and selling"
    return (f'The gold token trades ${px["premium"]:.2f} above spot, so I shift its VWAPs down by that (about ±$5–8 uncertain). '
            f'Taker buys {b}% = {tilt} this week. A mild lean, not a signal.')


def keys_say(out):
    sp, w = out["spot_xauusd"], out["walls"]
    bits = []
    cw, pw = w["call_wall_xauusd"], w["put_wall_xauusd"]
    if cw and pw:
        bits.append(f'Walls are the biggest option piles, ${abs(cw - sp):.0f} above and ${abs(sp - pw):.0f} below: too far for scalps.')
    vw, vm = out["vwap_week"], out["vwap_month"]
    if vw and vm:
        if sp < min(vw, vm):
            bits.append("Price is <b>below</b> both VWAPs (average prices): slightly bearish tilt, and rallies may stall at them.")
        elif sp > max(vw, vm):
            bits.append("Price is <b>above</b> both VWAPs (average prices): slightly bullish tilt, and dips may find buyers at them.")
        else:
            bits.append("Price is between the two VWAPs (average prices): mixed signal.")
    return " ".join(bits)


def keys_html(out):
    sp, flip, mp = out["spot_xauusd"], out["gamma_flip_xauusd"], out["max_pain_xauusd"]
    boxes = ""
    if flip:
        d, below = abs(flip - sp), sp < flip
        neutral = d <= 8 or abs(out["net_gex_usd_per_1pct"]) < 25e6
        if neutral:
            beh = (f"Price is sitting <b>right at the switch</b> (or total gamma is close to zero: {money(out['net_gex_usd_per_1pct'])} per 1%), "
                   "so there is <b>no clear speed edge</b> either way. The flip level moves a lot between runs here, so do not lean on it. "
                   "If price holds clearly above it, expect calmer moves; clearly below, faster ones.")
        elif below:
            beh = ("Expect <b>faster, bigger, jumpier</b> moves: dealer hedging tends to add fuel. Give trades more room and do not expect a slow grind. "
                   "A move back up through this level would suggest things calming down.")
        else:
            beh = ("Expect <b>calmer, slower, choppier</b> moves: dealer hedging tends to absorb dips and rallies. "
                   "A fall below this level would suggest moves getting faster.")
        boxes += (f'<div class="key"><h3>Gamma flip · the speed switch</h3><div class="kn">{flip:,.2f}</div>'
                  f'<div class="kl">Price is <b style="color:{"#ffb347" if below else "#6fd0ff"}">{"BELOW" if below else "ABOVE"}</b> it by ${d:.2f}{" (about at it)" if neutral else ""}</div>'
                  f'<div class="say" style="border:0;font-size:13.5px">{beh} '
                  f'<span class="meta">It tells you speed, not direction. Estimate based on an unverified assumption.</span></div></div>')
    if mp:
        d, above = abs(mp - sp), mp > sp
        boxes += (f'<div class="key"><h3>Max pain · weak magnet</h3><div class="kn">{mp:,.2f}</div>'
                  f'<div class="kl">Price is ${d:.2f} <b style="color:#6fd0ff">{"below" if above else "above"}</b> it</div>'
                  f'<div class="say" style="border:0;font-size:13.5px">The price where option sellers lose the least when the options expiring '
                  f'<b>{out["max_pain_expiry"]}</b> expire. Some traders think price drifts toward it, but the pull is <b>weak and unproven</b>. '
                  f'I would ignore it for scalps. The number resets after expiry.</div></div>')
    return f'<div class="keys">{boxes}</div>' if boxes else ""


def render_html(out):
    spot, s = out["spot_xauusd"], out["summary"]
    parts, placed = [], False
    full = out.get("full_table")
    cols = (["rank", "what", "zone", "dist", "pct", "big", "oi", "cp", "delta", "vol", "notional", "exp", "sig"] if full else
            ["rank", "what", "zone", "dist", "big", "oi", "cp", "delta", "exp", "sig"])
    wl = out.get("watch") or {}
    min_x = wl.get("min_x", 3.0)
    stars = {r["zone_lo"] for k in ("above", "below") for r in wl.get(k, [])[:1]}
    div = (f'<tr class="spot"><td colspan="{len(cols)}">● Gold is here: {spot:,.2f} &nbsp;·&nbsp; '
           f'above = possible ceilings (red) &nbsp;·&nbsp; below = possible floors (green)</td></tr>')
    for r in out["levels"]:
        if not placed and (r["zone_lo"] + r["zone_hi"]) / 2 <= spot:
            parts.append(div)
            placed = True
        parts.append(row_html(r, cols, min_x, r["kind"] == "strike" and r["zone_lo"] in stars))
    if not placed:
        parts.append(div)

    net, flip, w = out["net_gex_usd_per_1pct"], out["gamma_flip_xauusd"], out["walls"]
    cp, pp = s["call_premium_usd_approx"], s["put_premium_usd_approx"]
    cpct = 100 * cp / (cp + pp) if (cp + pp) else 50
    fx = lambda v: f"{v:,.2f}" if v else "n/a"
    sc = out.get("score")
    pill = ""
    if sc:
        k = "pos" if sc["value"] >= 0.35 else "neg" if sc["value"] <= -0.35 else "neu"
        pill = (f'<span class="pill p-{k}" title="swap component {sc["swap_component"]:+.2f}, options component '
                f'{sc["options_component"]:+.2f} (approximate formula)">Smart Money: {sc["label"]} ({sc["value"]:+.2f})</span>')

    cards = ""
    if out.get("cot"):
        c = out["cot"]

        def cotsay(name, b):
            pos = ("near the bottom of its 52-week range" if b["idx"] <= 25 else
                   "near the top of its 52-week range" if b["idx"] >= 75 else "mid-range for the year")
            if name == "Swap dealers":
                return (f'Banks hedging client trades, {pos}. Dealers are usually net short, so this is '
                        f'<b>not a bearish call</b>. Weekly data (as of {c["as_of"]}): background only.')
            lean = "net long = bullish lean" if b["net"] > 0 else "net short = bearish lean"
            ex = (" Crowded long: a risk if funds start selling." if b["idx"] >= 75 else
                  " Crowded short: a risk of a squeeze up." if b["idx"] <= 25 else " Not extreme either way.")
            wk = f' Last week funds {"added" if b["wk_chg"] > 0 else "cut"} {abs(b["wk_chg"]):,} contracts.'
            return f'Hedge funds: {lean}, {pos}.{ex}{wk} Weekly data: background only.'

        def cotcard(name, b):
            return (f'<div class="card"><h3>{name} · COT {c["as_of"]}</h3>'
                    f'<div class="big {"pos" if b["net"] > 0 else "neg"}">{b["net"]:+,}</div>'
                    f'<div class="meta">contracts net · ≈ {money(b["notional"])} <span class="chg {"gp" if b["wk_chg"] > 0 else "rn"}">wk {b["wk_chg"]:+,}</span></div>'
                    f'<div class="gauge"><i style="left:{b["idx"]}%"></i></div>'
                    f'<div class="meta">52-week range position {b["idx"]} · rank {b["pct52"]}</div>'
                    f'{say(cotsay(name, b))}</div>')
        cards += cotcard("Swap dealers", c["swap"]) + cotcard("Managed money", c["mm"])
    cards += (
        f'<div class="card"><h3>Options flow, latest session (approx. premium)</h3>'
        f'<div style="display:flex;justify-content:space-between"><b class="pos">CALL {money(cp)}</b><b class="neg">PUT {money(pp)}</b></div>'
        f'<div class="pbar"><span style="width:{cpct:.0f}%"></span><span style="width:{100 - cpct:.0f}%"></span></div>'
        f'<div class="meta">{cpct:.0f}% · {100 - cpct:.0f}% | P/C volume {s["pc_ratio_volume"]}</div>'
        f'<div style="margin-top:4px"><b class="{"pos" if net > 0 else "neg"}">Dealer {"long" if net > 0 else "short"} gamma</b> '
        f'<span class="meta">{money(net)} / 1%</span></div>'
        f'{say("Dollar value of puts vs calls traded last session. Puts bought after a drop are already in profit, which makes this look bearish. <b>Ignore this bar.</b> " + ("Gamma: close to neutral right now, so <b>no clear speed edge</b>." if (abs(net) < 25e6 or (flip and abs(flip - spot) <= 8)) else "Gamma: dealer hedging likely <b>speeds up moves</b> right now." if net < 0 else "Gamma: dealer hedging likely <b>calms moves down</b> right now."))}</div>')
    if out.get("cot"):
        cards += (f'<div class="card wide"><h3>COT · net positions, 52 weeks · <span style="color:#4aa3ff">swap dealers</span> · '
                  f'<span style="color:#f5a623">managed money</span></h3>{render_cot_chart(out["cot"]["series"])}'
                  f'{say("Blue = banks, orange = funds, over the last year. It changes over weeks, not hours, so use it as background only.")}</div>')
    cards += (f'<div class="card"><h3>Key levels</h3><div>Call wall <b class="up">{fx(w["call_wall_xauusd"])}</b> · Put wall '
              f'<b class="dn">{fx(w["put_wall_xauusd"])}</b></div><div>VWAP W {fx(out["vwap_week"])} · VWAP M {fx(out["vwap_month"])}</div>'
              f'<div class="meta">Max pain {fx(out["max_pain_xauusd"])} ({out["max_pain_expiry"]}) · Flip {fx(flip)}</div>'
              f'{say(keys_say(out))}</div>')
    if out.get("paxg"):
        px = out["paxg"]
        cards += (f'<div class="card"><h3>{html.escape(px["symbol"])} · 24/7 gold token</h3>'
                  f'<div>Premium to spot <b>{px["premium"]:+.2f}</b> ({px["premium_pct"]:+.2f}%)</div>'
                  f'<div>Week taker buys <b class="{"pos" if (px["week_buy_pct"] or 50) >= 50 else "neg"}">{px["week_buy_pct"]}%</b> '
                  f'<span class="meta">net {px["week_delta_oz"]:+,.0f} oz</span></div>'
                  f'<div class="meta">VWAPs shifted by the premium · last bar {px["last_bar_utc"]} UTC</div>'
                  f'{say(paxg_say(px))}</div>')
    gq = out.get("gld_quote", out["gld_price"])
    gnote = f' <span class="meta">(quote {gq:.2f})</span>' if abs(gq - out["gld_price"]) > 0.05 else ""
    strip = keys_html(out) + (f'<div class="strip">Call OI <b>{s["call_oi"]:,}</b> · Put OI <b>{s["put_oi"]:,}</b> · P/C OI <b>{s["pc_ratio_oi"]}</b> · '
             f'Gamma flip <b>{fx(flip)}</b> · Max pain <b>{fx(out["max_pain_xauusd"])}</b> · GLD <b>{out["gld_price"]:.2f}</b>{gnote} · '
             f'ETF × <b>{out["ratio"]:.3f}</b> <span class="meta">({out["ratio_src"]})</span> · expiries ≤ {out["max_dte"]}d</div>')
    head = "<tr>" + "".join(f"<th>{HEADS[k]}</th>" for k in cols) + "</tr>"
    dnote = f"OI Δ vs {out['delta_vs']}" if out["delta_vs"] else out["delta_status"]
    age = out.get("quote_age_min")
    anote = f" | GLD quote {age:.0f} min old" if age is not None else ""
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>XAUUSD levels</title><style>{CSS}</style>'
            f'<h1>Smart Money · Options Levels · XAUUSD ({html.escape(out["symbol"])} proxy){pill}</h1>'
            f'<div class="meta">Updated {out["generated_local"]}{anote} | CBOE delayed ~15 min | {dnote}</div>'
            f'{ratio_banner(out)}{plain_summary(out)}{watch_html(out["watch"], spot) if out.get("watch") else ""}'
            f'<div class="cards">{cards}</div>{strip}'
            f'{legend_html()}<div class="wrap"><div class="tbl"><table>{head}{"".join(parts)}</table></div>'
            f'{chart_html(out)}</div>'
            f'<div class="foot"><b>Small print.</b> "Contracts open" (open interest) = option contracts that exist and have not been closed; '
            f'it is a map of where traders have positioned, not a buy/sell signal. "How big" = a zone\'s contracts divided by the typical zone in this table '
            f'(the Rank shows the order, #1 = most). Big/medium thresholds: 7× and --watch-min (default 3×). "Far" in the watchlist = more than --scalp-range dollars away. '
            f'Each zone is one GLD strike wide (about $11 of gold) and the conversion to gold prices can be off by roughly $5–15, so treat edges as approximate. '
            f'Big ceiling / Big floor = my rule (zone at least 3× typical, or the biggest call/put level), not a trade signal. "Added overnight" = contracts added or removed since the last saved snapshot. '
            f'<b>Funds rows are an estimate</b>: the mid-price of a past COT week with a big Managed Money change, ±0.5%, not real fund orders. '
            f'<b>The Smart Money score is an approximate reconstruction</b> (0.7 × swap-dealer index + 0.5 × options premium skew), unvalidated; the options-flow card can be skewed by in-the-money puts after a drop, so I would ignore both. '
            f'The gamma / speed reading assumes market makers are long calls and short puts, which is not verified. Use --full-table for extra columns (%, traded today, size in $).</div>')



# --------------------------------------------------------------------- pine
def build_pine(out, rng=60.0, nmax=8, clusters_only=False):
    """Pine Script v6: a few merged, labelled levels near price (call/put walls, new OI, flip, max pain, VWAPs)."""
    ascii_ = lambda t: (str(t).replace('"', "'").replace("\\", "/").replace("\u2013", "-").replace("\u00b7", "-")
                        .encode("ascii", "replace").decode())
    spot = out["spot_xauusd"]
    min_x = (out.get("watch") or {}).get("min_x", 3.0)
    fk = lambda v: f"{v / 1000:.1f}K" if v >= 1000 else f"{int(v)}"
    prio = {"Call Wall": 0, "Put Wall": 0, "New CALL": 1, "New PUT": 1, "Big call pile": 2, "Big put pile": 2,
            "Max Pain": 3, "Gamma Flip": 4, "VWAP week": 5, "VWAP month": 5, "VWAP W + M": 5, "Week POC": 5, "Month POC": 5, "Week VAH": 6, "Week VAL": 6, "Month VAH": 6, "Month VAL": 6, "Funds buy (est.)": 6,
            "Funds sell (est.)": 6}
    RED, GREEN, PURPLE, YELLOW, BLUE, TEAL, PINK = "#ff5252", "#2ecc71", "#b79bff", "#f5c542", "#6fd0ff", "#4fe0d0", "#ff9ec7"
    cands, marks = [], []                        # cands: level lines with names; marks: dotted marker lines
    for r in out["levels"]:
        if r["kind"] == "strike":
            tg = r["tags"]
            names = [n for n in ("Call Wall", "Put Wall") if n in tg]
            newt = [n for n in ("New CALL", "New PUT") if n in tg]
            x = r["strength_x"]
            if newt and x >= min_x:
                names += newt
            if not names and x >= 7:
                names = ["Big call pile" if r["call_oi"] >= r["put_oi"] else "Big put pile"]
            if names:
                px_ = (r["zone_lo"] + r["zone_hi"]) / 2
                cands.append({"px": px_, "names": names, "col": RED if px_ > spot else GREEN, "strike": r,
                              "oi": f"#{r['rank']}  C {fk(r['call_oi'])} / P {fk(r['put_oi'])}"})
        elif r["kind"] == "funds":
            px_ = (r["zone_lo"] + r["zone_hi"]) / 2
            nm = "Funds buy (est.)" if "BUY" in r["tags"][0] else "Funds sell (est.)"
            cands.append({"px": px_, "names": [nm], "col": PINK, "strike": None, "oi": ""})
    vw, vm = out["vwap_week"], out["vwap_month"]
    msrc = [("Gamma Flip", out["gamma_flip_xauusd"], PURPLE), ("Max Pain", out["max_pain_xauusd"], YELLOW)]
    if vw and vm and abs(vw - vm) <= 6:
        msrc.append(("VWAP W + M", (vw + vm) / 2, BLUE))
    else:
        msrc += [("VWAP week", vw, BLUE), ("VWAP month", vm, TEAL)]
    for k_, v_ in ((out.get("paxg") or {}).get("profile") or {}).items():
        if out.get("show_profile", True) and k_.endswith("POC"):   # Pine keeps only the volume peaks, to stay uncluttered
            lab_ = {"Wk": "Week", "Mo": "Month"}[k_[:2]] + " " + k_[3:]
            msrc.append((lab_, v_, "#ffa94d" if k_.endswith("POC") else "#8aa0c8"))
    for nm, px_, col in msrc:
        if px_ and abs(px_ - spot) <= rng:
            marks.append({"px": float(px_), "names": [nm], "col": col, "strike": None, "oi": ""})
    items = sorted([c for c in cands if abs(c["px"] - spot) <= rng] + marks, key=lambda c: c["px"])
    clusters = []
    for it in items:                             # merge labels that sit within $6 of each other
        if clusters and it["px"] - clusters[-1][0]["px"] <= 6:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    outc = []
    for cl in clusters:
        names = sorted({n for c in cl for n in c["names"]}, key=lambda n: prio.get(n, 9))
        first = min(cl, key=lambda c: prio.get(c["names"][0], 9))
        host = next((c for c in cl if c["strike"] is not None), None) or first
        outc.append({"px": host["px"], "names": names, "col": first["col"],
                     "oi": host["oi"], "dist": abs(host["px"] - spot)})
    outc = sorted(sorted(outc, key=lambda c: c["dist"])[:nmax], key=lambda c: -c["px"])
    if clusters_only:                            # used by the website: same merged levels as the Pine script
        PAX = ("VWAP week", "VWAP month", "VWAP W + M", "Week POC", "Month POC")
        prem_ = float((out.get("paxg") or {}).get("premium") or 0.0)
        return [{"px": round(c["px"], 2), "names": c["names"], "col": c["col"], "oi": c["oi"],
                 "pax": all(n_ in PAX for n_ in c["names"]), "raw": round(c["px"] + prem_, 2),
                 "dotted": all(n in ("Gamma Flip", "Max Pain", "VWAP week", "VWAP month", "VWAP W + M", "Week POC", "Month POC")
                               for n in c["names"])} for c in outc]
    keep = lambda c: any(abs(c["px"] - o["px"]) <= 6 for o in outc)
    lines = [c for c in cands if abs(c["px"] - spot) <= rng and keep(c)]
    mlines = [m for m in marks if keep(m)]
    rw = out.get("ratio_warn")
    note = ascii_(rw[1]) if rw and rw[0] != "ok" else ""
    code = [f"""//@version=6
// Generated by gld_levels.py at {ascii_(out['generated_local'])}, gold spot {spot:.2f}.
// Paste into TradingView > Pine Editor > Save > Add to chart. Use on OANDA:XAUUSD.
// Levels are static: run the script again and paste the new version to refresh them.
indicator("GLD Options Levels", overlay=true, max_labels_count=60)

showLevels  = input.bool(true, "Show option levels (solid lines)")
showMarkers = input.bool(true, "Show flip / max pain / VWAP (dotted lines)")
showLabels  = input.bool(true, "Show labels")
labelOff    = input.int(4, "Label offset (bars to the right)", minval=0, maxval=100)
staleHours  = input.float(2.0, "Mark as STALE after (hours)", minval=0.1)
moveWarn    = input.float(30.0, "Warn when price moved this many $ since the run", minval=1.0)

genMs   = {out.get('generated_ms', 0)}
genSpot = {spot:.2f}
"""]
    for i, c in enumerate(lines):
        nm = ascii_(" + ".join(c["names"]))
        code.append(f'plot(showLevels ? {c["px"]:.2f} : na, "{nm} {c["px"]:.0f}", color={c["col"]}, linewidth=1, style=plot.style_linebr)')
    for m in mlines:
        code.append(f'plot(showMarkers ? {m["px"]:.2f} : na, "{ascii_(m["names"][0])} {m["px"]:.0f}", color={m["col"]}, linewidth=1, style=plot.style_circles)')
    code.append("""
var array<label> lbs = array.new<label>()
var table info = table.new(position.top_right, 1, 1)

if barstate.islast
    for lb in lbs
        label.delete(lb)
    array.clear(lbs)""")
    code.append("    if showLabels")
    if not outc:
        code.append("        float unused = na")
    for c in outc:
        txt = ascii_("  ".join(x for x in (" + ".join(c["names"]), f"{c['px']:,.2f}", c["oi"]) if x))
        code.append(f'        array.push(lbs, label.new(bar_index + labelOff, {c["px"]:.2f}, "{txt}", style=label.style_label_left, '
                    f'color=color.new(#0b0f1a, 10), textcolor={c["col"]}, size=size.normal))')
    code.append(f"""    ageH = (timenow - genMs) / 3600000.0
    moved = close - genSpot
    stale = ageH > staleHours or math.abs(moved) > moveWarn
    msg = ""
    if ageH > staleHours
        msg += "STALE: levels are " + str.tostring(ageH, "#.#") + " h old. Run the script again.\\n"
    if math.abs(moved) > moveWarn
        msg += "Price moved " + str.tostring(moved, "#.#") + " since the run.\\n"
    msg += "Levels from {ascii_(out['generated_local'])}" + ("{"\\n" + note if note else ""}")
    table.cell(info, 0, 0, msg, text_color=color.white, bgcolor=stale ? color.new(color.red, 30) : color.new(color.gray, 60), text_size=size.small)
""")
    return "\n".join(code)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GLD")
    ap.add_argument("--spot", help="spot XAUUSD price (OANDA on TradingView)")
    ap.add_argument("--ratio", type=float, help="fixed XAUUSD-per-GLD-share ratio (use when GLD quote is stale)")
    ap.add_argument("--vwap-week", type=float, help="weekly anchored VWAP from TradingView")
    ap.add_argument("--vwap-month", type=float, help="monthly anchored VWAP from TradingView")
    ap.add_argument("--paxg", action="store_true", help="VWAP W/M, flow and funds-zone prices from Binance PAXGUSDT")
    ap.add_argument("--paxg-symbol", default="PAXGUSDT")
    ap.add_argument("--max-dte", type=int, default=45, help="only expiries within N days (try 2 for near-term only)")
    ap.add_argument("--band", type=float, default=0.20)
    ap.add_argument("--rows", type=int, default=7, help="max option rows above and below price")
    ap.add_argument("--range", type=float, default=150.0, help="max $ distance from spot to list")
    ap.add_argument("--wall-range", type=float, default=250.0, help="max $ distance from spot for put/call wall search")
    ap.add_argument("--watch-min", type=float, default=3.0, help="min strength (x) for a level to appear in the watchlist")
    ap.add_argument("--scalp-range", type=float, default=30.0, help="levels closer than this ($) are 'close enough for a scalp'")
    ap.add_argument("--new-min", type=int, default=500, help="contracts of OI added to tag New CALL/PUT")
    ap.add_argument("--pine-range", type=float, default=60.0, help="Pine: only levels within this many $ of price")
    ap.add_argument("--pine-max", type=int, default=8, help="Pine: max number of labelled levels")
    ap.add_argument("--web-range", type=float, default=120.0, help="levels within this many $ of price go to the website")
    ap.add_argument("--web-max", type=int, default=14, help="max levels on the website chart")
    ap.add_argument("--no-profile", action="store_true", help="hide the volume-profile levels (POC / value area) from PAXG")
    ap.add_argument("--copy-pine", action="store_true", help="also copy the Pine Script to the clipboard (Windows)")
    ap.add_argument("--full-table", action="store_true", help="show extra table columns (%%, traded today, size in $)")
    ap.add_argument("--no-cot", action="store_true", help="skip CFTC COT cards and funds zones")
    ap.add_argument("--cot-id", default=COT_DATASET, help="CFTC dataset id if the default stops working")
    ap.add_argument("--funds-lookback", type=int, default=52, help="COT weeks to look back for funds zones")
    ap.add_argument("--funds-top", type=int, default=6, help="keep the N biggest weekly changes")
    ap.add_argument("--funds-min", type=int, default=5000, help="min weekly Managed Money net change (contracts)")
    ap.add_argument("--funds-center", choices=["mid", "vwap"], default="mid",
                    help="funds zone centre: week mid-range (default, matches the original) or week VWAP")
    ap.add_argument("--ratio-live", action="store_true", help="force spot/GLD ratio even when the GLD quote is stale")
    ap.add_argument("--tz", type=float, default=7.0)
    ap.add_argument("--out", default="output")
    ap.add_argument("--snap-dir", default="snapshots")
    ap.add_argument("--from-file")
    a = ap.parse_args()

    gld_px, raw, qtime = fetch_chain(a.symbol, a.from_file)
    spot_gold = get_spot_gold(a.spot)
    now = datetime.now(timezone.utc)
    now_et = utc_to_et(now)
    live_today = now_et.hour < 16          # today's expiring options still trade until 16:00 ET
    age = quote_age_min(qtime, now)
    stale = age is not None and age > 20
    if age is not None:
        print(f"GLD quote age: {age:.0f} min (last trade {qtime} ET)")

    paxg = None
    if a.paxg:
        try:
            print("Fetching PAXGUSDT from Binance...", flush=True)
            paxg = paxg_stats(a.paxg_symbol, spot_gold, now, quote_utc(qtime))
            if a.vwap_week is None:
                a.vwap_week = paxg["vwap_week_spot"]
            if a.vwap_month is None:
                a.vwap_month = paxg["vwap_month_spot"]
            print(f"PAXG: last {paxg['last']:.2f} (premium to spot {paxg['premium']:+.2f}) | "
                  f"VWAP W {a.vwap_week} | VWAP M {a.vwap_month} | last bar {paxg['last_bar_utc']} UTC")
        except Exception as e:
            print(f"PAXG skipped: {e}. Pass --vwap-week/--vwap-month manually.", file=sys.stderr)

    Path(a.snap_dir).mkdir(exist_ok=True)
    rfile = Path(a.snap_dir) / "ratio.json"
    ratio_live = spot_gold / gld_px
    n_read = 0
    if age is None:   # no quote time in the data: fall back to the clock
        h = now.hour + now.minute / 60
        stale = now.weekday() >= 5 or not (13.5 <= h <= 21.0)
    if a.ratio:
        ratio, ratio_src = a.ratio, "fixed by --ratio"
    elif a.ratio_live:
        ratio, ratio_src = ratio_live, "live spot / GLD (forced)"
    elif not stale and 10.5 < ratio_live < 11.5:
        # GLD's quote lags live spot by ~15 min, so one reading is noisy: use the median of recent fresh readings
        stamp = now.strftime("%Y-%m-%d %H:%M")
        hist = []
        try:
            for x in json.loads(rfile.read_text()).get("readings", []):
                if (now.replace(tzinfo=None) - datetime.strptime(x["t"], "%Y-%m-%d %H:%M")).total_seconds() < 6 * 3600:
                    hist.append(x)
        except Exception:
            pass
        hist = (hist + [{"t": stamp, "ratio": round(ratio_live, 4)}])[-7:]
        n_read = len(hist)
        ratio = float(np.median([x["ratio"] for x in hist]))
        ratio_src = (f"median of {len(hist)} fresh readings in the last 6h; this run alone {ratio_live:.3f}"
                     if len(hist) > 1 else "live spot / GLD (first fresh reading)")
        rfile.write_text(json.dumps({"ratio": ratio, "saved_utc": stamp, "gld": gld_px, "spot": spot_gold,
                                     "readings": hist}))
        print(f"Ratio this run {ratio_live:.3f}; using {ratio:.3f} ({len(hist)} fresh reading(s) in the last 6h: "
              f"{', '.join(f'{x['ratio']:.3f}' for x in hist)}). Saved.")
    else:
        saved = None
        try:
            saved = json.loads(rfile.read_text())
            if (now.replace(tzinfo=None) - datetime.strptime(saved["saved_utc"], "%Y-%m-%d %H:%M")).days > 30:
                saved = None
        except Exception:
            saved = None
        if saved:
            ratio, ratio_src = float(saved["ratio"]), f"last fresh-quote reading ({saved['saved_utc'][:10]})"
        else:
            ratio, ratio_src = DEFAULT_RATIO, f"default {DEFAULT_RATIO}, approximate - run once in US hours to calibrate"
        print(f"GLD quote is stale or pre-market -> using ratio {ratio:.3f} ({ratio_src}). "
              f"Spot/GLD would give {ratio_live:.3f}, which is unreliable right now.")
    gld_quote = gld_px
    gld_px = spot_gold / ratio   # GLD-equivalent price of today's gold: used for GEX, flip and level conversion

    today = pd.Timestamp(now_et.date())            # trading date in New York
    today_s = today.strftime("%Y-%m-%d")
    allp = parse(raw)
    snap = allp[allp["expiry"] >= today][KEY + ["open_interest"]].reset_index(drop=True)
    delta_vs, d, had_prev = find_delta(a.snap_dir, today_s, snap)
    if d is not None:
        snap["oi_delta"] = d
    delta_status = ("OI Δ: OI identical to the last snapshot (CBOE updates OI overnight; try again after ~18:00 your time)"
                    if had_prev else "OI Δ: no earlier snapshot yet (run again on another day)")
    Path(a.snap_dir).mkdir(exist_ok=True)
    snap[KEY + ["open_interest"]].to_csv(Path(a.snap_dir) / f"{today_s}.csv", index=False)

    df = prepare(allp, today, a.max_dte, a.band, gld_px)
    if not live_today:
        df = df[df["dte"] >= 1].reset_index(drop=True)   # today's options already expired
    if df.empty:
        sys.exit("No options left after filters - check --max-dte / --band or the data.")
    if d is not None:
        df = df.merge(snap[KEY + ["oi_delta"]], on=KEY, how="left")
        df["oi_delta"] = df["oi_delta"].fillna(0.0)

    flip_gld, net = gamma_flip(df, gld_px), net_gex(df, gld_px)
    cand = df[(df["open_interest"] > 0) & ((df["dte"] >= 1) | ((df["dte"] == 0) & live_today))]
    near_exp = (cand if len(cand) else df)["expiry"].min()
    mp_gld = max_pain(df[df["expiry"] == near_exp])
    flip_x = flip_gld * ratio if flip_gld else None
    mp_x = mp_gld * ratio if mp_gld else None

    cot_sum, funds = None, []
    if not a.no_cot:
        try:
            print("Fetching CFTC COT...", flush=True)
            cot = fetch_cot(a.cot_id)
            cot_sum = cot_summary(cot, spot_gold)
            sw_, mm_ = cot_sum["swap"], cot_sum["mm"]
            print(f"COT {cot_sum['as_of']}: swap {sw_['net']:+,} (range pos {sw_['idx']}, rank {sw_['pct52']}) | "
                  f"MM {mm_['net']:+,} (range pos {mm_['idx']}, rank {mm_['pct52']})")
            try:
                if paxg:
                    print("Fetching PAXG daily history for funds zones...", flush=True)
                    px, shift = paxg_daily(a.paxg_symbol), -paxg["premium"]
                else:
                    print("Fetching gold futures prices (yfinance) for funds zones...", flush=True)
                    px = fetch_px()
                    shift = spot_gold - float(px["Close"].iloc[-1])
                funds = funds_zones(cot, px, shift, spot_gold, a.funds_lookback, a.funds_top, a.funds_min,
                                    center_mode=a.funds_center)
            except Exception as e:
                print(f"Funds zones skipped (price data): {e}", file=sys.stderr)
        except Exception as e:
            print(f"COT skipped: {e}", file=sys.stderr)

    summ = summarise(df)
    score = smart_score(cot_sum, summ) if cot_sum else None
    specials = [("Gamma flip", flip_x), ("Max pain", mp_x), ("VWAP W", a.vwap_week), ("VWAP M", a.vwap_month)]
    if paxg and paxg.get("profile") and not a.no_profile:
        specials += list(paxg["profile"].items())
    levels, chart, walls, skipped = build_table(df, gld_px, ratio, spot_gold, specials, funds, a.rows, a.range, a.new_min, a.wall_range)
    out = {
        "symbol": a.symbol, "max_dte": a.max_dte, "quote_age_min": age,
        "generated_local": (now + timedelta(hours=a.tz)).strftime("%Y-%m-%d %H:%M") + f" (UTC{a.tz:+g})",
        "gld_price": gld_px, "gld_quote": gld_quote, "spot_xauusd": round(spot_gold, 2), "ratio": ratio, "ratio_src": ratio_src,
        "net_gex_usd_per_1pct": net, "gamma_flip_xauusd": round(flip_x, 2) if flip_x else None,
        "gamma_regime": ("positive gamma (hedging dampens moves)" if net > 0 else "negative gamma (hedging amplifies moves)"),
        "max_pain_xauusd": round(mp_x, 2) if mp_x else None, "max_pain_expiry": near_exp.strftime("%Y-%m-%d"),
        "vwap_week": a.vwap_week, "vwap_month": a.vwap_month, "walls": walls, "cot": cot_sum, "paxg": paxg,
        "score": score, "delta_vs": delta_vs, "delta_status": delta_status, "summary": summ, "levels": levels, "chart": chart,
    }
    out["watch"] = build_watchlist(out, a.watch_min, a.scalp_range)
    out["full_table"] = a.full_table
    out["show_profile"] = not a.no_profile
    out["generated_ms"] = int(now.timestamp() * 1000)
    if a.ratio:
        out["ratio_warn"] = ("ok", f"Ratio fixed by you at {ratio:.3f}.")
    elif n_read == 0 and ratio_src.startswith("default"):
        out["ratio_warn"] = ("bad", "No fresh price ratio yet: strike levels (ceilings, floors, flip, max pain) can be off by $10–50. "
                                    "Run again during US hours (after 20:50 your time). VWAPs are not affected.")
    elif n_read == 0:
        out["ratio_warn"] = ("warn", "Using an older saved ratio (the GLD quote is stale): strike levels can be off by $10–20. "
                                     "Run again during US hours (after 20:50 your time). VWAPs are not affected.")
    elif n_read < 3:
        out["ratio_warn"] = ("warn", f"Only {n_read} fresh reading(s) so far: strike levels can still shift by about $10 between runs. "
                                     "Run it 2–3 more times, 10–15 minutes apart, to steady them. VWAPs are not affected.")
    else:
        out["ratio_warn"] = ("ok", f"Ratio steady: median of {n_read} fresh readings. Strike levels are good to roughly ±$5–10.")
    out["spot_source"] = SPOT_SOURCE
    out["web_levels"] = build_pine(out, a.web_range, a.web_max, clusters_only=True)
    outdir = Path(a.out)
    outdir.mkdir(exist_ok=True)
    (outdir / "levels.json").write_text(json.dumps(out, indent=2))
    (outdir / "dashboard.html").write_text(render_html(out), encoding="utf-8")
    pine = build_pine(out, a.pine_range, a.pine_max)
    (outdir / "levels.pine").write_text(pine, encoding="utf-8")
    if a.copy_pine:
        try:
            import subprocess
            subprocess.run(["clip"], input=pine.encode("ascii", "replace"), check=True)
            print("Pine Script copied to the clipboard: paste it into the TradingView Pine editor.")
        except Exception as e:
            print(f"Could not copy to the clipboard ({e}); open {outdir / 'levels.pine'} instead.")

    print(f"GLD {gld_px:.2f} | XAUUSD {spot_gold:.2f} | ratio {ratio:.3f} ({ratio_src})")
    print(f"Net GEX {money(net)} per 1% | {out['gamma_regime']}")
    print(f"Gamma flip: {out['gamma_flip_xauusd']} | Max pain ({out['max_pain_expiry']}): {out['max_pain_xauusd']}")
    print(f"Call wall: {walls['call_wall_xauusd']} | Put wall: {walls['put_wall_xauusd']}")
    if score:
        print(f"Smart Money score (approx.): {score['value']:+.2f} {score['label']}")
    print(f"OI change vs {delta_vs}" if delta_vs else delta_status)
    print(f"COT: {'ok' if cot_sum else 'not loaded'} | funds zones shown: {len([x for x in levels if x['kind'] == 'funds'])}"
          f"{f' ({skipped} beyond --range)' if skipped else ''}")
    wl = out["watch"]
    for lab, key in (("Nearest strong above", "above"), ("Nearest strong below", "below")):
        r = wl[key][0] if wl[key] else None
        print(f"{lab}: {r['zone_lo']:,.2f} - {r['zone_hi']:,.2f} ({'+' if key == 'above' else '-'}${r['dist']:.2f}, x{r['strength_x']:.1f})"
              if r else f"{lab}: none at x{a.watch_min:g}+ in range")
    print(f"Saved {outdir / 'levels.json'}, {outdir / 'dashboard.html'} and {outdir / 'levels.pine'}")


if __name__ == "__main__":
    main()
