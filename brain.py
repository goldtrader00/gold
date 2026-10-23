#!/usr/bin/env python3
"""
brain.py - "Brain" for the gold levels site.

What it does (every run, about every 15 minutes, 24 hours a day on weekdays):
  1. Memory levels: prices where 1-hour swing highs/lows turned 3+ times in the last ~60 days.
  2. Touch tracking: replays PAXG 1-minute candles since the last run against the levels that were
     on the site AT THAT TIME (levels.json as it was then), so nothing is judged with hindsight.
       touch  = price comes into a level's zone (level +/- ZONE) from outside
       HELD   = price then moves TARGET dollars past the near edge, away from the level (a bounce)
       BROKE  = price moves TARGET dollars past the far edge first (went through)
       neither within 4 hours = timeout (not counted)
  3. Scores every current level (Brain %) from the real touch results, by source (Put OI, Memory,
     VWAP ...), session, and whether the touch is with or against the 4-hour trend.
  4. Day meter: today's move vs a normal day's range (range day / trend day).

Output: docs/brain.json (results + the small state it needs for the next run).
All prices inside are PAXG token prices ("raw"); the page converts them to XAUUSD the same way
it converts the candles, so lines, candles and the ladder always agree.

First run: with --backfill-git it replays the git history of docs/levels.json, so the Brain starts
with every touch since the site went live instead of from zero.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

VERSION = 1
BINANCE = ["https://data-api.binance.vision", "https://api.binance.com"]
SYMBOL = "PAXGUSDT"
NY = ZoneInfo("America/New_York")

ZONE = 3.0          # a level's zone = level +/- $3
TARGET = 5.0        # bounce / break distance beyond the zone edge
TIMEOUT_MIN = 240   # unresolved after 4 hours -> not counted
HALF_LIFE_D = 30.0  # older touches count less (weight halves every 30 days)
MERGE = 3.0         # levels closer than $3 are treated as one zone
KEEP_DAYS = 180
TAG_MIN_N = 50      # no BUY/SELL tags until the Brain has at least this many results
TAG_PCT, CONS_PCT = 0.62, 0.55

SPECIAL_TAGS = {"Gamma flip", "Max pain", "VWAP W", "VWAP M", "Wk POC", "Mo POC",
                "Wk VAL", "Wk VAH", "Mo VAL", "Mo VAH"}
NICE = {"Call OI": "Call OI", "Put OI": "Put OI", "Call Wall": "Call Wall", "Put Wall": "Put Wall",
        "New CALL": "New CALL", "New PUT": "New PUT", "Funds BUY": "Funds buy (est.)",
        "Funds SELL": "Funds sell (est.)", "VWAP W": "VWAP week", "VWAP M": "VWAP month",
        "POC": "Volume POC", "Gamma flip": "Gamma flip", "Max pain": "Max pain", "Memory": "Memory level"}
SESS = [(0, 7, "Asia"), (7, 12, "London"), (12, 17, "New York"), (17, 24, "Late US")]


# --------------------------------------------------------------------------- helpers
def now_ms():
    return int(time.time() * 1000)


def session_of(ms):
    h = datetime.fromtimestamp(ms / 1000, timezone.utc).hour
    for a, b, n in SESS:
        if a <= h < b:
            return n
    return "Late US"


def gold_closed(ms):
    """Same rule as the page: OANDA gold is shut Fri 17:00 - Sun 18:00 New York, and 17:00-18:00 daily."""
    d = datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(NY)
    wd, h = d.weekday(), d.hour   # Mon=0
    return (wd == 4 and h >= 17) or wd == 5 or (wd == 6 and h < 18) or (h == 17 and wd in (0, 1, 2, 3))


def weight(age_ms):
    return 0.5 ** (max(age_ms, 0) / 86400000 / HALF_LIFE_D)


# --------------------------------------------------------------------------- data
def fetch_klines(interval, start_ms, end_ms=None, candles_file=None):
    """Binance PAXGUSDT klines -> DataFrame[t(ms), o,h,l,c,v]. candles_file: offline testing."""
    if candles_file:
        d = json.load(open(candles_file))[interval]
        df = pd.DataFrame(d, columns=["t", "o", "h", "l", "c", "v"])
        df = df[df.t >= start_ms]
        return df[df.t <= end_ms] if end_ms else df
    err = None
    for base in BINANCE:
        try:
            rows, t = [], start_ms
            while True:
                p = {"symbol": SYMBOL, "interval": interval, "startTime": t, "limit": 1000}
                if end_ms:
                    p["endTime"] = end_ms
                r = requests.get(f"{base}/api/v3/klines", params=p, timeout=20)
                r.raise_for_status()
                b = r.json()
                if not b:
                    break
                rows += b
                t = b[-1][6] + 1
                if len(b) < 1000:
                    break
            df = pd.DataFrame([[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])]
                               for x in rows], columns=["t", "o", "h", "l", "c", "v"])
            return df
        except Exception as e:  # try the next host
            err = e
    raise RuntimeError(f"Binance klines failed: {err}")


def git_snapshots(path="docs/levels.json"):
    """Every committed version of levels.json (oldest first) -> [(generated_ms, dict)]."""
    try:
        hs = subprocess.run(["git", "log", "--reverse", "--format=%H", "--", path],
                            capture_output=True, text=True, check=True).stdout.split()
    except Exception as e:
        print("git history not available:", e)
        return []
    out = []
    for h in hs:
        try:
            d = json.loads(subprocess.run(["git", "show", f"{h}:{path}"], capture_output=True,
                                          text=True, check=True).stdout)
            if d.get("generated_ms"):
                out.append((int(d["generated_ms"]), d))
        except Exception:
            pass
    out.sort(key=lambda x: x[0])
    return out


# --------------------------------------------------------------------------- levels
def memory_levels(h1, asof_ms, ref_px, days=60, lr=3, tol=5.0, min_n=3, band=200.0, top=12):
    """Prices where 1h swing highs/lows clustered 3+ times in the last `days` before asof (no lookahead)."""
    sub = h1[(h1.t < asof_ms) & (h1.t >= asof_ms - days * 86400000)].reset_index(drop=True)
    if len(sub) < 2 * lr + 5:
        return []
    hi, lo, ts = sub.h.values, sub.l.values, sub.t.values
    piv = []
    for i in range(lr, len(sub) - lr):
        w_hi, w_lo = hi[i - lr:i + lr + 1], lo[i - lr:i + lr + 1]
        if hi[i] == w_hi.max() and (w_hi == hi[i]).sum() == 1:
            piv.append((hi[i], ts[i]))
        if lo[i] == w_lo.min() and (w_lo == lo[i]).sum() == 1:
            piv.append((lo[i], ts[i]))
    piv.sort()
    clusters = []
    for p, t in piv:
        if clusters and abs(p - np.mean(clusters[-1][0])) <= tol:
            clusters[-1][0].append(p)
            clusters[-1][1].append(t)
        else:
            clusters.append(([p], [t]))
    out = []
    for ps, tt in clusters:
        c = float(np.mean(ps))
        if len(ps) >= min_n and abs(c - ref_px) <= band:
            out.append({"raw": round(c, 2), "count": len(ps), "last_ms": int(max(tt))})
    out.sort(key=lambda m: (-m["count"], abs(m["raw"] - ref_px)))
    return sorted(out[:top], key=lambda m: m["raw"])


def comps_from_levels(lv, mem):
    """levels.json (+ memory levels) -> list of single-source components in token ('raw') prices."""
    prem = float((lv.get("paxg") or {}).get("premium") or 0.0)
    comps = []

    def add(key, tags, xau, raw=None, pax=False, extra=None):
        raw = raw if raw is not None else xau + prem
        c = {"key": key, "tags": tags, "raw": round(raw, 2), "px": round(xau, 2), "pax": pax}
        if extra:
            c.update(extra)
        comps.append(c)

    for l in lv.get("levels") or []:
        tags = [t for t in l.get("tags", []) if t not in SPECIAL_TAGS]
        if l.get("kind") == "strike" and tags:
            mid = (l["zone_lo"] + l["zone_hi"]) / 2
            add(f"S:{l.get('gld_strike')}", tags, mid, extra={
                "calls": l.get("call_oi"), "puts": l.get("put_oi"), "x": l.get("strength_x")})
        elif l.get("kind") == "funds" and tags:
            mid = (l["zone_lo"] + l["zone_hi"]) / 2
            add(f"F:{l.get('expiry', '')}:{round(mid)}", tags, mid)
    pax = lv.get("paxg") or {}
    for k, nm in (("vwap_week_token", "VWAP W"), ("vwap_month_token", "VWAP M")):
        if pax.get(k):
            add(nm, [nm], pax[k] - prem, raw=pax[k], pax=True)
    prof = pax.get("profile") or {}
    for nm in ("Wk POC", "Mo POC"):
        if prof.get(nm):
            add(nm, ["POC"], prof[nm], raw=prof[nm] + prem, pax=True)
    if lv.get("gamma_flip_xauusd"):
        add("GF", ["Gamma flip"], lv["gamma_flip_xauusd"])
    if lv.get("max_pain_xauusd"):
        add("MP", ["Max pain"], lv["max_pain_xauusd"])
    for m in mem or []:
        add(f"M:{round(m['raw'])}", ["Memory"], m["raw"] - prem, raw=m["raw"], pax=True,
            extra={"count": m["count"]})
    return comps


def group(comps):
    """Merge components within MERGE dollars into zones."""
    comps = sorted(comps, key=lambda c: c["raw"])
    out = []
    for c in comps:
        if out and c["raw"] - out[-1]["parts"][-1]["raw"] <= MERGE:
            out[-1]["parts"].append(c)
        else:
            out.append({"parts": [c]})
    for g in out:
        ps = g["parts"]
        g["raw"] = round(float(np.mean([p["raw"] for p in ps])), 2)
        g["keys"] = sorted({p["key"] for p in ps})
        g["tags"] = sorted({t for p in ps for t in p["tags"]})
    return out


# --------------------------------------------------------------------------- trend / day
def h4_trend_series(h1):
    """4h EMA20 vs EMA50 -> list of (bar_close_ms, 'up'|'down')."""
    if h1.empty:
        return []
    d = h1.copy()
    d["b"] = (d.t // (4 * 3600000)) * (4 * 3600000)
    h4 = d.groupby("b").agg(c=("c", "last")).reset_index()
    e20 = h4.c.ewm(span=20, adjust=False).mean()
    e50 = h4.c.ewm(span=50, adjust=False).mean()
    return [(int(b + 4 * 3600000), "up" if a > z else "down") for b, a, z in zip(h4.b, e20, e50)]


def trend_at(series, ms):
    tr = "flat"
    for close_ms, v in series:   # last completed 4h bar before ms
        if close_ms <= ms:
            tr = v
        else:
            break
    return tr


def trading_day_start(ms):
    """Gold trading day starts 18:00 New York."""
    d = datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone(NY)
    s = d.replace(hour=18, minute=0, second=0, microsecond=0)
    if d < s:
        s -= timedelta(days=1)
    return int(s.astimezone(timezone.utc).timestamp() * 1000)


def day_meter(h1, m1, now):
    d0 = trading_day_start(now)
    hist = h1[h1.t < d0].copy()
    hist["day"] = hist.t.apply(trading_day_start)
    rng = hist[~hist.t.apply(gold_closed)].groupby("day").agg(h=("h", "max"), l=("l", "min"), n=("t", "size"))
    rng = rng[rng.n >= 12]
    adr = float((rng.h - rng.l).tail(14).mean()) if len(rng) else None
    today = m1[m1.t >= d0]
    if today.empty:
        today = h1[h1.t >= d0]
    if today.empty:
        return {"start_ms": d0, "adr": adr}
    return {"start_ms": d0, "adr": round(adr, 2) if adr else None, "open": float(today.o.iloc[0]),
            "high": float(today.h.max()), "low": float(today.l.min()), "last": float(today.c.iloc[-1])}


# --------------------------------------------------------------------------- replay
def replay(bars, schedule, h1, trend_ser, state):
    """bars: 1m DataFrame (open market only). schedule: [(from_ms, levels_dict)] sorted.
    Adds finished touches to state['touches'], keeps unfinished ones in state['pending']."""
    pend = state["pending"]
    touches = state["touches"]
    si, zones, mem_day, mem = -1, [], None, []
    prev_c = state.get("last_close")
    T = bars.t.values
    H, L, C = bars.h.values, bars.l.values, bars.c.values
    for i in range(len(bars)):
        t = int(T[i])
        # 1) resolve open touches with this bar
        still = []
        for p in pend:
            lvl, s = p["L"], p["side"]
            up_hit = H[i] >= lvl + ZONE + TARGET
            dn_hit = L[i] <= lvl - ZONE - TARGET
            if s == "S":
                p["mfe"] = max(p["mfe"], H[i] - (lvl + ZONE))
                p["mae"] = max(p["mae"], (lvl + ZONE) - L[i])
                res = "B" if dn_hit else ("H" if up_hit else None)
            else:
                p["mfe"] = max(p["mfe"], (lvl - ZONE) - L[i])
                p["mae"] = max(p["mae"], H[i] - (lvl - ZONE))
                res = "B" if up_hit else ("H" if dn_hit else None)
            if res is None and t - p["t"] >= TIMEOUT_MIN * 60000:
                res = "T"
            if res:
                p.update(res=res, rt=t, mfe=round(p["mfe"], 2), mae=round(p["mae"], 2))
                touches.append(p)
            else:
                still.append(p)
        pend[:] = still
        # 2) which levels were live at this minute
        changed = False
        while si + 1 < len(schedule) and schedule[si + 1][0] <= t:
            si += 1
            changed = True
        day = t // 86400000
        if day != mem_day:
            mem_day = day
            mem = memory_levels(h1, day * 86400000, C[i])
            changed = True
        if changed and si >= 0:
            zones = group(comps_from_levels(schedule[si][1], mem))
        # 3) new touches: previous close outside the zone, this bar reaches into it
        if prev_c is not None and zones:
            busy = {k for p in pend for k in p["keys"]}
            for z in zones:
                lvl = z["raw"]
                if busy.intersection(z["keys"]):
                    continue
                side = None
                if prev_c > lvl + ZONE and L[i] <= lvl + ZONE:
                    side = "S"   # came down onto it: support test
                elif prev_c < lvl - ZONE and H[i] >= lvl - ZONE:
                    side = "R"   # came up into it: resistance test
                if side:
                    tr = trend_at(trend_ser, t)
                    wt = "flat" if tr == "flat" else ("with" if (tr == "up") == (side == "S") else "against")
                    pend.append({"t": t, "L": lvl, "side": side, "keys": z["keys"], "tags": z["tags"],
                                 "sess": session_of(t), "trend": wt, "mfe": 0.0, "mae": 0.0})
        prev_c = float(C[i])
    state["last_close"] = prev_c
    if len(bars):
        state["last_ts"] = int(T[-1])


# --------------------------------------------------------------------------- stats / scores
def rate(rows, now, prior_p=0.5, k=10.0):
    w_h = sum(weight(now - r["t"]) for r in rows if r["res"] == "H")
    w_n = sum(weight(now - r["t"]) for r in rows)
    held = sum(1 for r in rows if r["res"] == "H")
    return {"p": (w_h + prior_p * k) / (w_n + k), "raw": (held / len(rows)) if rows else None,
            "held": held, "n": len(rows), "wn": w_n}


def build_stats(touches, now):
    done = [r for r in touches if r["res"] in ("H", "B")]
    base = rate(done, now)
    p0 = base["p"]
    out = {"overall": base, "last20": rate(done[-20:], now, p0, 0) if done else None,
           "timeouts": sum(1 for r in touches if r["res"] == "T")}

    def by(fn, names=None):
        d = {}
        for r in done:
            for v in fn(r):
                d.setdefault(v, []).append(r)
        return {k: rate(v, now, p0, 20) for k, v in d.items() if names is None or k in names}

    out["tag"] = by(lambda r: r["tags"])
    out["sess"] = by(lambda r: [r["sess"]])
    out["trend"] = by(lambda r: [r["trend"]])
    out["side"] = by(lambda r: [r["side"]])
    keys = {}
    for r in done:
        for k in r["keys"]:
            keys.setdefault(k, []).append(r)
    out["key"] = {k: {"held": sum(1 for r in v if r["res"] == "H"), "n": len(v),
                      "p": rate(v, now, p0, 8)["p"]} for k, v in keys.items()}
    return out


def score_zone(z, side, st, cur_sess, cur_trend):
    p0 = st["overall"]["p"]
    factors = []
    for t in z["tags"]:
        s = st["tag"].get(t)
        if s and s["n"] >= 3:
            factors.append((NICE.get(t, t), s["p"] - p0))
    held = n = 0
    kp = []
    for k in z["keys"]:
        s = st["key"].get(k)
        if s:
            held += s["held"]
            n += s["n"]
            kp.append(s["p"] - p0)
    if kp:
        factors.append((f"held {held}/{n} here", float(np.mean(kp))))
    if cur_trend in ("up", "down"):
        wt = "with" if (cur_trend == "up") == (side == "S") else "against"
        s = st["trend"].get(wt)
        if s and s["n"] >= 5:
            factors.append((f"{wt} 4h trend", s["p"] - p0))
    # (session is shown in the stats, but not used in a zone's score: it says nothing about the zone)
    # tag factors overlap (one zone has several tags) -> average them, then add the rest
    tagf = [f for f in factors if not f[0].startswith(("held", "with", "against")) and "session" not in f[0]]
    other = [f for f in factors if f not in tagf]
    score = p0 + (float(np.mean([f[1] for f in tagf])) if tagf else 0.0) + sum(f[1] for f in other) * 0.5
    score = min(0.95, max(0.05, score))
    return score, held, n, [(nm, round(v * 100)) for nm, v in factors if abs(v) >= 0.01]


def zones_out(lv, mem, st, last_raw, cur_sess, cur_trend):
    zs = group(comps_from_levels(lv, mem))
    ready = st["overall"]["n"] >= TAG_MIN_N
    out = []
    for z in zs:
        lvl = z["raw"]
        pos = "in" if abs(last_raw - lvl) <= ZONE else ("above" if lvl > last_raw else "below")
        side = "R" if pos == "above" else "S" if pos == "below" else None
        sc, held, n, f = score_zone(z, side or "S", st, cur_sess, cur_trend)
        scS, _, _, fS = score_zone(z, "S", st, cur_sess, cur_trend)   # if price is above it (support)
        scR, _, _, fR = score_zone(z, "R", st, cur_sess, cur_trend)   # if price is below it (resistance)
        if pos == "in":
            tag = "IN ZONE"
        elif not ready:
            tag = "RESISTANCE" if side == "R" else "SUPPORT"
        elif sc >= TAG_PCT:
            tag = "SELL" if side == "R" else "BUY"
        elif sc >= CONS_PCT:
            tag = "CONSIDER SELL" if side == "R" else "CONSIDER BUY"
        else:
            tag = "RESISTANCE" if side == "R" else "SUPPORT"
        parts = z["parts"]
        calls = sum(p.get("calls") or 0 for p in parts)
        puts = sum(p.get("puts") or 0 for p in parts)
        xs = [p["x"] for p in parts if p.get("x")]
        allpax = all(p["pax"] for p in parts)
        out.append({"raw": lvl, "px": round(float(np.mean([p["px"] for p in parts])), 2), "pax": allpax,
                    "tags": z["tags"], "names": [NICE.get(t, t) for t in z["tags"]], "keys": z["keys"],
                    "score": round(sc * 100) if ready else None, "tag": tag, "held": held, "n": n,
                    "factors": f, "sS": round(scS * 100), "sR": round(scR * 100), "fS": fS, "fR": fR,
                    "calls": calls or None, "puts": puts or None,
                    "x": max(xs) if xs else None,
                    "mem_count": max([p.get("count") or 0 for p in parts]) or None})
    return sorted(out, key=lambda z: -z["raw"])


def insight(st, day, cur_trend, cur_sess):
    o = st["overall"]
    lines = []
    if o["n"] == 0:
        return ["The Brain is still collecting its first level touches. Scores appear after about "
                f"{TAG_MIN_N} finished touches."]
    lines.append(f"Levels held {round(o['raw'] * 100)}% of {o['n']} finished touches "
                 f"(bounce ${TARGET:.0f} before breaking ${TARGET:.0f}; a coin flip would be 50%).")
    if o["n"] < TAG_MIN_N:
        lines.append(f"Still learning: BUY/SELL tags switch on after {TAG_MIN_N} results.")
    tg = [(NICE.get(k, k), v) for k, v in st["tag"].items() if v["n"] >= 8]
    if tg:
        best = max(tg, key=lambda x: x[1]["p"])
        worst = min(tg, key=lambda x: x[1]["p"])
        lines.append(f"Best source so far: {best[0]} ({best[1]['held']}/{best[1]['n']} held). "
                     f"Weakest: {worst[0]} ({worst[1]['held']}/{worst[1]['n']}).")
    w, a = st["trend"].get("with"), st["trend"].get("against")
    if w and a and w["n"] >= 5 and a["n"] >= 5:
        lines.append(f"With the 4h trend: {w['held']}/{w['n']} held. Against it: {a['held']}/{a['n']}.")
    s = st["sess"].get(cur_sess)
    if s and s["n"] >= 5:
        lines.append(f"{cur_sess} session touches: {s['held']}/{s['n']} held.")
    if cur_trend in ("up", "down"):
        lines.append(f"4h trend right now: {cur_trend.upper()} - support bounces are 'with trend' in an uptrend, "
                     f"resistance rejections in a downtrend.")
    return lines


def stats_out(st):
    def slim(d):
        return {k: {"held": v["held"], "n": v["n"], "pct": round(v["raw"] * 100) if v["raw"] is not None else None}
                for k, v in d.items()}
    o = st["overall"]
    return {"overall": {"held": o["held"], "n": o["n"], "pct": round(o["raw"] * 100) if o["raw"] is not None else None},
            "last20": ({"held": st["last20"]["held"], "n": st["last20"]["n"],
                        "pct": round(st["last20"]["raw"] * 100)} if st["last20"] else None),
            "timeouts": st["timeouts"], "tag": slim({NICE.get(k, k): v for k, v in st["tag"].items()}),
            "sess": slim(st["sess"]), "trend": slim(st["trend"]),
            "side": slim({"Support (came down)" if k == "S" else "Resistance (came up)": v
                          for k, v in st["side"].items()})}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="docs/levels.json")
    ap.add_argument("--out", default="docs/brain.json")
    ap.add_argument("--backfill-git", action="store_true", help="first run: replay git history of levels.json")
    ap.add_argument("--candles-file", help="offline test data instead of Binance")
    ap.add_argument("--now-ms", type=int, help="testing: pretend it is this time")
    a = ap.parse_args()
    now = a.now_ms or now_ms()

    lv_now = json.load(open(a.levels))
    outp = Path(a.out)
    old = json.loads(outp.read_text()) if outp.exists() else {}
    state = old.get("state") if old.get("version") == VERSION else None

    schedule = []
    if state is None:
        state = {"pending": [], "touches": [], "last_ts": None, "last_close": None, "levels_ms": None}
        if a.backfill_git:
            schedule = git_snapshots(a.levels)
            print(f"backfill: {len(schedule)} saved versions of levels.json")
        start = schedule[0][0] if schedule else now - 30 * 60000
        if lv_now.get("generated_ms") and (not schedule or schedule[-1][0] < lv_now["generated_ms"]):
            schedule.append((int(lv_now["generated_ms"]), lv_now))
    else:
        start = (state["last_ts"] + 60000) if state.get("last_ts") else now - 30 * 60000
        prev = old.get("levels_snapshot")   # the levels that were live since the last run
        if prev:
            schedule.append((0, prev))
        if lv_now.get("generated_ms") and lv_now["generated_ms"] != (prev or {}).get("generated_ms"):
            schedule.append((int(lv_now["generated_ms"]), lv_now))
        elif not prev:
            schedule.append((0, lv_now))
    start = max(start, now - 14 * 86400000)

    h1 = fetch_klines("1h", now - 75 * 86400000, candles_file=a.candles_file)
    m1 = fetch_klines("1m", start, now, candles_file=a.candles_file)
    m1 = m1[m1.t + 60000 <= now]                               # finished candles only
    if len(m1):
        m1 = m1[~m1.t.apply(gold_closed).astype(bool)].reset_index(drop=True)
    else:
        m1 = pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    trend_ser = h4_trend_series(h1)
    print(f"1m candles to replay: {len(m1)}  (from {datetime.fromtimestamp(start / 1000, timezone.utc):%d/%m %H:%M} UTC)")

    replay(m1, schedule, h1, trend_ser, state)
    cut = now - KEEP_DAYS * 86400000
    state["touches"] = [r for r in state["touches"] if r["t"] >= cut][-6000:]

    st = build_stats(state["touches"], now)
    last_raw = float(m1.c.iloc[-1]) if len(m1) else float(h1.c.iloc[-1])
    cur_trend = trend_at(trend_ser, now)
    cur_sess = session_of(now)
    mem = memory_levels(h1, now, last_raw)
    day = day_meter(h1, m1, now)
    res = {
        "version": VERSION, "generated_ms": now,
        "rules": {"zone": ZONE, "target": TARGET, "timeout_min": TIMEOUT_MIN, "half_life_days": HALF_LIFE_D,
                  "tag_pct": round(TAG_PCT * 100), "consider_pct": round(CONS_PCT * 100), "tag_min_n": TAG_MIN_N, "ready": st["overall"]["n"] >= TAG_MIN_N},
        "last_raw": last_raw, "trend_4h": cur_trend, "session": cur_sess,
        "memory": mem, "day": day,
        "zones": zones_out(lv_now, mem, st, last_raw, cur_sess, cur_trend),
        "stats": stats_out(st),
        "recent": [{k: r[k] for k in ("t", "L", "side", "tags", "sess", "trend", "res", "mfe", "mae")}
                   for r in state["touches"][-25:]][::-1],
        "pending": len(state["pending"]),
        "insight": insight(st, day, cur_trend, cur_sess),
        "levels_snapshot": lv_now,
        "state": state,
    }
    outp.write_text(json.dumps(res, separators=(",", ":")))
    o = res["stats"]["overall"]
    print(f"touches: {o['n']} finished ({o['held']} held, {o['pct']}%), {len(state['pending'])} open, "
          f"memory levels: {len(mem)}, zones: {len(res['zones'])}, 4h trend: {cur_trend}")


if __name__ == "__main__":
    main()
