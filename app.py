"""
GEX Engine — XAUUSD / US500 / US100
Streamlit app replacing the Colab notebook.
Run: streamlit run app.py
Deploy free: streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import requests
import time
import warnings
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date, timedelta

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEX Engine",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #080c10;
    color: #e8eaed;
}
h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; }

.gex-header {
    background: linear-gradient(135deg, #0d1f0d 0%, #0a1628 100%);
    border: 1px solid #1a3a1a;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
}
.gex-header h1 {
    font-size: 2rem;
    margin: 0 0 4px 0;
    background: linear-gradient(90deg, #00e676, #00bcd4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.gex-header p { color: #8a9bb0; margin: 0; font-size: 0.9rem; }

.paste-block {
    background: #0d1117;
    border: 1px solid #1e3a1e;
    border-left: 3px solid #00e676;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #c9d1d9;
    white-space: pre;
    overflow-x: auto;
}
.paste-block-us {
    border-left-color: #00bcd4;
}
.level-card {
    background: #0d1f2d;
    border: 1px solid #1a3050;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.level-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #15253a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.level-row:last-child { border-bottom: none; }
.level-label { color: #8a9bb0; }
.level-val { color: #e8eaed; font-weight: 600; }
.lvl-green { color: #00e676; }
.lvl-red   { color: #ff5252; }
.lvl-cyan  { color: #00bcd4; }
.lvl-yellow{ color: #ffd740; }
.lvl-purple{ color: #ce93d8; }
.lvl-orange{ color: #ffab40; }

.section-divider {
    border-top: 1px solid #1a3050;
    margin: 4px 0 2px 0;
}
.section-header {
    color: #4a6070;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 6px 0 2px 0;
}

.source-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin-bottom: 16px;
}
.badge-cme  { background: #0d2a0d; color: #00e676; border: 1px solid #00e676; }
.badge-gld  { background: #2a1a00; color: #ffd740; border: 1px solid #ffd740; }

.regime-pos { color: #00e676; font-weight: 700; }
.regime-neg { color: #ff5252; font-weight: 700; }
.regime-purple { color: #ce93d8; font-weight: 700; }
.regime-orange { color: #ffab40; font-weight: 700; }

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00e676, #00bcd4);
    color: #000;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 14px;
    cursor: pointer;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c853, #0097a7);
}
.stNumberInput > div > div > input {
    background: #0d1f2d;
    border: 1px solid #1a3050;
    color: #e8eaed;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    text-align: center;
}
.footer {
    text-align: center;
    color: #3d5268;
    font-size: 0.75rem;
    padding: 24px 0 8px 0;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE        = 0.05
GC_MULTIPLIER         = 100
EQ_MULTIPLIER         = 100
CME_PRODUCT_ID        = 444
CME_OPTION_PRODUCT_ID = 193
CME_EXCHANGE          = "COMEX"
CME_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/html, */*;q=0.9",
    "Referer": "https://www.cmegroup.com/markets/metals/precious/gold.settlements.options.html",
}

# ── MATHS ─────────────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d1 - sig*np.sqrt(T))

def bs_put(S, K, T, r, sig):
    if T <= 0 or sig <= 0: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_gamma(S, K, T, r, sig):
    if T <= 1e-6 or sig <= 1e-6 or S <= 0 or K <= 0: return 0.0
    try:
        d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
        return norm.pdf(d1) / (S * sig * np.sqrt(T))
    except: return 0.0

def bs_call_delta(S, K, T, r, sig):
    if T <= 1e-6 or sig <= 1e-6 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    try:
        d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
        return norm.cdf(d1)
    except: return 0.0

def bs_put_delta(S, K, T, r, sig):
    if T <= 1e-6 or sig <= 1e-6 or S <= 0 or K <= 0:
        return -1.0 if S < K else 0.0
    try:
        d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
        return norm.cdf(d1) - 1.0
    except: return 0.0

def bs_vanna(S, K, T, r, sig):
    """
    Vanna = dDelta/dVol = dVega/dSpot
    = (d2 / sig) * norm.pdf(d1)
    Sign convention: positive vanna = delta increases when vol rises (OTM calls / ITM puts).
    Dealers short calls → long vanna → buy spot when vol drops, sell when vol rises.
    """
    if T <= 1e-6 or sig <= 1e-6 or S <= 0 or K <= 0: return 0.0
    try:
        d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        return (d2 / sig) * norm.pdf(d1)
    except: return 0.0

def bs_charm(S, K, T, r, sig):
    """
    Charm = dDelta/dTime (per calendar day, annualised basis)
    Negative charm on calls = delta decays toward 0 for OTM as time passes.
    Net charm exposure tells you the direction of daily dealer re-hedging flow.
    Returned per-year; divide by 365 for per-day in caller.
    """
    if T <= 1e-6 or sig <= 1e-6 or S <= 0 or K <= 0: return 0.0
    try:
        d1 = (np.log(S/K) + (r + .5*sig**2)*T) / (sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        # dDelta/dT for a call (positive = delta rises toward expiry for ITM calls)
        charm = -norm.pdf(d1) * (2*r*T - d2*sig*np.sqrt(T)) / (2*T*sig*np.sqrt(T))
        return charm
    except: return 0.0

def implied_vol(price, S, K, T, r, opt="call"):
    intr = max(S-K,0.) if opt=="call" else max(K-S,0.)
    if price <= intr + 1e-6 or T <= 0: return 0.15
    try:
        fn = bs_call if opt=="call" else bs_put
        return brentq(lambda s: fn(S,K,T,r,s) - price, 1e-4, 10.)
    except: return 0.15

def biz_day(offset=0):
    d = date.today() - timedelta(days=offset)
    while d.weekday() >= 5: d -= timedelta(days=1)
    return d

def dte(expiry_str):
    try:
        e = datetime.strptime(expiry_str[:10], "%Y-%m-%d").date()
        return max((e - date.today()).days, 1)
    except: return 20

# ── CME FETCHERS ──────────────────────────────────────────────────────────────
def _parse_json(data):
    rows = []
    items = data if isinstance(data, list) else data.get("settlements", data.get("rows", []))
    for item in items:
        try:
            s   = float(str(item.get("strikePrice", item.get("strike",0))).replace(",",""))
            coi = int(str(item.get("callOpenInt",0)).replace(",","") or 0)
            poi = int(str(item.get("putOpenInt", 0)).replace(",","") or 0)
            cs  = float(str(item.get("callSettle",0)).replace(",","") or 0)
            ps  = float(str(item.get("putSettle", 0)).replace(",","") or 0)
            civ = float(str(item.get("callIV",    0)).replace(",","") or 0) / 100
            piv = float(str(item.get("putIV",     0)).replace(",","") or 0) / 100
            if s > 0:
                rows.append({"strike":s,"call_oi":coi,"put_oi":poi,
                             "call_settle":cs,"put_settle":ps,"call_iv":civ,"put_iv":piv})
        except: continue
    return pd.DataFrame(rows) if len(rows) > 5 else None

def _parse_html(html):
    try:
        tables = pd.read_html(html)
        for tbl in tables:
            if len(tbl) > 10 and tbl.shape[1] >= 5:
                fc = tbl.iloc[:,0].astype(str)
                if fc.str.replace(",","").str.match(r"^\d{3,5}(\.\d+)?$").sum() > 10:
                    tbl.columns = [" ".join(str(c) for c in col).strip()
                                   if isinstance(col,tuple) else str(col)
                                   for col in tbl.columns]
                    tbl = tbl.fillna(0)
                    recs = []
                    for _, row in tbl.iterrows():
                        v = list(row)
                        try:
                            recs.append({
                                "strike":      float(str(v[0]).replace(",","")),
                                "call_oi":     int(float(str(v[2]).replace(",","").replace("-","0") or 0)) if len(v)>2  else 0,
                                "call_settle": float(str(v[7]).replace(",","").replace("-","0") or 0)      if len(v)>7  else 0,
                                "put_oi":      int(float(str(v[10]).replace(",","").replace("-","0") or 0))if len(v)>10 else 0,
                                "put_settle":  float(str(v[15]).replace(",","").replace("-","0") or 0)     if len(v)>15 else 0,
                                "call_iv":0.0,"put_iv":0.0,
                            })
                        except: continue
                    df = pd.DataFrame(recs)
                    df = df[df["strike"] > 0]
                    if len(df) > 5: return df
    except: pass
    return None

def _try_cme_json(td):
    ds = td.strftime("%Y%m%d")
    url = (f"https://www.cmegroup.com/CmeWS/mvc/Settlements/futures/options"
           f"/tradeDate/{ds}/productId/{CME_PRODUCT_ID}"
           f"/exchange/{CME_EXCHANGE}/optionProductId/{CME_OPTION_PRODUCT_ID}")
    try:
        s = requests.Session()
        s.get("https://www.cmegroup.com/markets/metals/precious/gold.settlements.options.html",
              headers=CME_HEADERS, timeout=12)
        time.sleep(0.8)
        r = s.get(url, headers={**CME_HEADERS,"Accept":"application/json"}, timeout=15)
        if r.status_code == 200:
            try:    return _parse_json(r.json())
            except: return _parse_html(r.text)
    except: pass
    return None

def _try_cme_html(td):
    ds = td.strftime("%Y%m%d")
    url = (f"https://www.cmegroup.com/CmeWS/mvc/Settlements/futures/options"
           f"/tradeDate/{ds}/productId/{CME_PRODUCT_ID}"
           f"/exchange/{CME_EXCHANGE}/optionProductId/{CME_OPTION_PRODUCT_ID}")
    try:
        s = requests.Session()
        s.headers.update(CME_HEADERS)
        r = s.get(url, timeout=15)
        if r.status_code == 200 and "<table" in r.text.lower():
            df = _parse_html(r.text)
            if df is not None and len(df) > 5: return df
    except: pass
    return None

def _gld_proxy(gc_spot, authoritative_spot=None):
    """
    authoritative_spot: user-entered XAUUSD price. When provided, strikes are
    scaled to this price rather than GC=F futures (which can lag by $100+).
    """
    GOLD_PROXIES = ["GLD", "IAU", "SGOL"]
    for ticker_sym in GOLD_PROXIES:
        try:
            t        = yf.Ticker(ticker_sym)
            hist     = t.history(period="5d")
            if hist.empty: continue
            etf_spot = float(hist["Close"].dropna().iloc[-1])
            # Use authoritative user-entered spot for scaling if provided
            scale    = (authoritative_spot if authoritative_spot else gc_spot) / etf_spot
            exps     = t.options
            if not exps: continue
            best_expiry, best_oi, best_calls, best_puts = None, 0, None, None
            for exp in exps[:6]:
                try:
                    ch    = t.option_chain(exp)
                    total = ch.calls["openInterest"].sum() + ch.puts["openInterest"].sum()
                    if total > best_oi:
                        best_oi, best_expiry = total, exp
                        best_calls, best_puts = ch.calls, ch.puts
                except: continue
            if best_calls is None or best_oi < 100:
                continue
            df = pd.merge(
                best_calls[["strike","openInterest","impliedVolatility"]].rename(
                    columns={"openInterest":"call_oi","impliedVolatility":"call_iv"}),
                best_puts[["strike","openInterest","impliedVolatility"]].rename(
                    columns={"openInterest":"put_oi","impliedVolatility":"put_iv"}),
                on="strike", how="outer"
            ).fillna(0)
            df["call_settle"] = 0.0
            df["put_settle"]  = 0.0
            df["strike"]      = (df["strike"] * scale).round(1)
            spot_out = authoritative_spot if authoritative_spot else gc_spot
            return df, spot_out, best_expiry, etf_spot, scale, ticker_sym
        except: continue
    raise RuntimeError("All gold proxies failed (GLD, IAU, SGOL). Try again in a few minutes.")

# ── CORE COMPUTATION ──────────────────────────────────────────────────────────
def enrich(df, spot, multiplier, days=30):
    T  = max(days / 365.0, 1/365)
    df = df.copy()
    civ_l, piv_l = [], []
    cg_l, pg_l   = [], []
    cd_l, pd_l   = [], []
    cv_l, pv_l   = [], []   # vanna
    cc_l, pc_l   = [], []   # charm

    for _, row in df.iterrows():
        K   = row["strike"]
        civ = (row["call_iv"] if row.get("call_iv",0) > 0.01
               else implied_vol(row.get("call_settle",0), spot, K, T, RISK_FREE_RATE, "call")
               if row.get("call_settle",0) > 0.01 else 0.15)
        piv = (row["put_iv"] if row.get("put_iv",0) > 0.01
               else implied_vol(row.get("put_settle",0), spot, K, T, RISK_FREE_RATE, "put")
               if row.get("put_settle",0) > 0.01 else civ if civ > 0.01 else 0.15)
        civ_l.append(civ); piv_l.append(piv)
        cg_l.append(bs_gamma(spot, K, T, RISK_FREE_RATE, civ))
        pg_l.append(bs_gamma(spot, K, T, RISK_FREE_RATE, piv))
        cd_l.append(bs_call_delta(spot, K, T, RISK_FREE_RATE, civ))
        pd_l.append(bs_put_delta(spot, K, T, RISK_FREE_RATE, piv))
        cv_l.append(bs_vanna(spot, K, T, RISK_FREE_RATE, civ))
        pv_l.append(bs_vanna(spot, K, T, RISK_FREE_RATE, piv))
        cc_l.append(bs_charm(spot, K, T, RISK_FREE_RATE, civ))
        pc_l.append(bs_charm(spot, K, T, RISK_FREE_RATE, piv))

    df["call_iv_c"] = civ_l; df["put_iv_c"] = piv_l
    df["call_gamma"]= cg_l;  df["put_gamma"] = pg_l
    df["call_delta"]= cd_l;  df["put_delta"] = pd_l
    df["call_vanna"]= cv_l;  df["put_vanna"] = pv_l
    df["call_charm"]= cc_l;  df["put_charm"] = pc_l

    # GEX: gamma × OI × multiplier × S²
    df["call_gex"]  =  df["call_gamma"] * df["call_oi"] * multiplier * spot**2
    df["put_gex"]   = -df["put_gamma"]  * df["put_oi"]  * multiplier * spot**2
    df["net_gex"]   =  df["call_gex"]   + df["put_gex"]

    # DEX: delta × OI × multiplier × S
    df["call_dex"]  =  df["call_delta"] * df["call_oi"] * multiplier * spot
    df["put_dex"]   =  df["put_delta"]  * df["put_oi"]  * multiplier * spot
    df["net_dex"]   =  df["call_dex"]   + df["put_dex"]

    # VEX (Vanna Exposure): vanna × OI × multiplier × S
    # Dealers short calls → positive call_vanna position → buy spot when IV drops
    # Net VEX > 0 = vol-down is bullish (dealers buy spot); < 0 = vol-down is bearish
    df["call_vex"]  =  df["call_vanna"] * df["call_oi"] * multiplier * spot
    df["put_vex"]   = -df["put_vanna"]  * df["put_oi"]  * multiplier * spot
    df["net_vex"]   =  df["call_vex"]   + df["put_vex"]

    # CEX (Charm Exposure): per-day delta units from dealer perspective
    # Dealers SHORT calls  → charm exposure = -(call_charm × OI) → they LOSE delta each day on OTM calls
    # Dealers LONG puts    → charm exposure = -(put_charm × OI)  → put charm is negative for OTM puts
    # Net positive CEX = dealers need to BUY spot each day to stay hedged (bullish daily pressure)
    # Net negative CEX = dealers need to SELL spot each day (bearish daily pressure)
    df["call_cex"]  = -df["call_charm"] * df["call_oi"] * multiplier * spot / 365
    df["put_cex"]   =  df["put_charm"]  * df["put_oi"]  * multiplier * spot / 365
    df["net_cex"]   =  df["call_cex"]   + df["put_cex"]

    return df

def levels(df, spot, label="", days=30):
    # Use a tighter initial window; fall back to progressively wider ranges
    # to guarantee enough strikes for meaningful level computation
    filt = None
    for pct in [0.10, 0.20, 0.30, 0.50, 1.0]:
        mask = (df["strike"] >= spot*(1-pct)) & (df["strike"] <= spot*(1+pct))
        candidate = df[mask].sort_values("strike").reset_index(drop=True)
        if len(candidate) >= 10:
            filt = candidate
            break
    if filt is None or filt.empty:
        filt = df.sort_values("strike").reset_index(drop=True)

    # ── GEX levels ────────────────────────────────────────────────────────────
    call_wall = float(filt.loc[filt["call_gex"].idxmax(), "strike"])
    put_wall  = float(filt.loc[filt["put_gex"].idxmin(),  "strike"])

    cum = filt["net_gex"].cumsum().values
    crossings = np.where(np.diff(np.sign(cum)))[0]
    if len(crossings) > 0:
        ci  = crossings[len(crossings)//2]
        ci1 = min(ci+1, len(filt)-1)
        s0, s1 = filt.iloc[ci]["strike"], filt.iloc[ci1]["strike"]
        v0, v1 = float(cum[ci]), float(cum[ci1])
        gflip = (s0 - v0*(s1-s0)/(v1-v0)) if (v1-v0) != 0 else s0
    else:
        gflip = float(filt.iloc[np.argmin(np.abs(cum))]["strike"])
    # Clamp GEX Flip between Put Wall and Call Wall — structural requirement
    lo_wall = min(put_wall, call_wall)
    hi_wall = max(put_wall, call_wall)
    gflip = float(np.clip(gflip, lo_wall, hi_wall))

    # ── DEX levels ────────────────────────────────────────────────────────────
    call_delta_wall = float(filt.loc[filt["call_dex"].idxmax(), "strike"])
    put_delta_wall  = float(filt.loc[filt["put_dex"].idxmin(),  "strike"])

    dex_cum = filt["net_dex"].cumsum().values
    dex_crossings = np.where(np.diff(np.sign(dex_cum)))[0]
    if len(dex_crossings) > 0:
        ci  = dex_crossings[len(dex_crossings)//2]
        ci1 = min(ci+1, len(filt)-1)
        s0, s1 = filt.iloc[ci]["strike"], filt.iloc[ci1]["strike"]
        v0, v1 = float(dex_cum[ci]), float(dex_cum[ci1])
        dflip = (s0 - v0*(s1-s0)/(v1-v0)) if (v1-v0) != 0 else s0
    else:
        dflip = float(filt.iloc[np.argmin(np.abs(dex_cum))]["strike"])
    lo_dwall = min(put_delta_wall, call_delta_wall)
    hi_dwall = max(put_delta_wall, call_delta_wall)
    dflip = float(np.clip(dflip, lo_dwall, hi_dwall))

    # ── VEX levels (Vanna Flip) ───────────────────────────────────────────────
    vex_cum = filt["net_vex"].cumsum().values
    vex_crossings = np.where(np.diff(np.sign(vex_cum)))[0]
    if len(vex_crossings) > 0:
        ci  = vex_crossings[len(vex_crossings)//2]
        ci1 = min(ci+1, len(filt)-1)
        s0, s1 = filt.iloc[ci]["strike"], filt.iloc[ci1]["strike"]
        v0, v1 = float(vex_cum[ci]), float(vex_cum[ci1])
        vanna_flip = (s0 - v0*(s1-s0)/(v1-v0)) if (v1-v0) != 0 else s0
    else:
        vanna_flip = float(filt.iloc[np.argmin(np.abs(vex_cum))]["strike"])

    # ── Max Pain ──────────────────────────────────────────────────────────────
    # Use filtered strikes — full df includes far-OTM strikes with noise OI
    # that pull Max Pain away from the tradeable zone
    sa  = filt["strike"].values
    coi = filt["call_oi"].values.astype(float)
    poi = filt["put_oi"].values.astype(float)
    lss = [np.sum(np.maximum(sa-s,0)*coi)+np.sum(np.maximum(s-sa,0)*poi) for s in sa]
    max_pain = float(sa[np.argmin(lss)])

    # ── Scalar exposures ──────────────────────────────────────────────────────
    total_net_gex = float(filt["net_gex"].sum())
    total_net_dex = float(filt["net_dex"].sum())
    total_net_vex = float(filt["net_vex"].sum())
    total_net_cex = float(filt["net_cex"].sum())   # per-day delta units

    # ── Regime labels ─────────────────────────────────────────────────────────
    # GEX regime: spot vs GEX Flip tells you whether dealers are in pinning or
    # trend-amplifying mode.
    gex_positive = spot > gflip
    regime = "POSITIVE ▲" if gex_positive else "NEGATIVE ▼"

    # DEX bias: net dealer delta tells you directional hedging pressure.
    dex_positive = total_net_dex > 0
    dex_bias = "BULLISH ▲" if dex_positive else "BEARISH ▼"

    # Vanna regime — per SpotGamma / SqueezeMetrics convention:
    # Net VEX > 0 means dealers are net long vanna (sold more calls than puts on vanna basis).
    # When IV FALLS  → dealers must SELL spot to stay hedged → bearish pressure from vol crush.
    # When IV RISES  → dealers must BUY spot → provides a floor during vol spikes.
    # Net VEX < 0 means dealers net short vanna.
    # When IV FALLS  → dealers BUY spot → bullish (vol crush = squeeze fuel).
    # When IV RISES  → dealers SELL spot → crash acceleration.
    # The document frames this as: "Price Up + IV Down = bullish Vanna squeeze"
    # That condition = spot above vanna flip (net VEX crossed negative) + IV falling.
    # We label by the DOMINANT condition: which IV scenario is dangerous/supportive.
    vex_positive = total_net_vex > 0
    if vex_positive:
        # Dealers net long vanna → IV drop forces them to sell → vol crush is bearish
        # IV spike forces them to buy → vol spike provides support
        vanna_regime = "VOL-CRUSH BEARISH ▼"
    else:
        # Dealers net short vanna → IV drop forces them to buy → vol crush is bullish (squeeze fuel)
        # IV spike forces them to sell → crash accelerator
        vanna_regime = "VOL-CRUSH BULLISH ▲"

    # Charm regime: DTE-weighted.
    # Charm is only reliable signal within 21 DTE. Beyond that, noise dominates.
    # sign already correct from enrich(): positive net_cex = dealers buy spot daily.
    cex_positive = total_net_cex > 0
    if days >= 21:
        charm_flow = "BULLISH BLEED ▲" if cex_positive else "BEARISH BLEED ▼"
        charm_reliable = False   # flag for combined engine
    else:
        charm_flow = "BULLISH BLEED ▲" if cex_positive else "BEARISH BLEED ▼"
        charm_reliable = True

    # ── Combined Regime Engine ────────────────────────────────────────────────
    # Combines GEX + DEX + Vanna into a single intraday condition + strategy.
    # This is the four-quadrant model from the framework document.
    #
    # GEX+ DEX+ → Stable Bullish  → buy dips, fade resistance
    # GEX+ DEX- → Stable Bearish  → sell rallies, fade support
    # GEX- DEX+ → Squeeze Risk    → momentum longs, breakout bias
    # GEX- DEX- → Crash/Trend     → momentum shorts, avoid catching bottoms
    #
    # Vanna adds acceleration signal on top.
    # Charm adds intraday time-decay direction (afternoon bias) when DTE < 21.

    if gex_positive and dex_positive:
        intraday_condition = "STABLE BULLISH"
        strategy           = "Buy pullbacks · Fade resistance · Avoid chasing"
        condition_cls      = "regime-pos"
    elif gex_positive and not dex_positive:
        intraday_condition = "STABLE BEARISH"
        strategy           = "Sell rallies · Fade bounces · Avoid breakout longs"
        condition_cls      = "regime-neg"
    elif not gex_positive and dex_positive:
        intraday_condition = "SQUEEZE RISK"
        strategy           = "Momentum longs · Breakout bias · Expect acceleration"
        condition_cls      = "regime-orange"
    else:  # GEX- DEX-
        intraday_condition = "CRASH / TREND"
        strategy           = "Momentum shorts · Avoid catching bottoms · Trail stops"
        condition_cls      = "regime-neg"

    # Vanna modifier: adds or removes conviction from the base condition
    if not vex_positive and dex_positive:
        vanna_note = "⚡ VOL CRUSH = SQUEEZE FUEL"   # dealers buy on IV drop
    elif vex_positive and not dex_positive:
        vanna_note = "⚡ VOL CRUSH = SELL PRESSURE"  # dealers sell on IV drop
    elif not vex_positive and not dex_positive:
        vanna_note = "⚠ VOL SPIKE = CRASH ACCELERATOR"
    else:
        vanna_note = "⚠ VOL SPIKE FINDS SUPPORT"

    # Charm intraday modifier (only show when reliable)
    if charm_reliable:
        charm_note = "📈 AFTERNOON MELT-UP BIAS" if cex_positive else "📉 AFTERNOON FADE BIAS"
    else:
        charm_note = f"⏱ CHARM: {'↑' if cex_positive else '↓'} (low weight — {days}+ DTE)"

    return {
        # GEX
        "Call Wall":            round(call_wall,          1),
        "Put Wall":             round(put_wall,           1),
        "GEX Flip":             round(gflip,              1),
        # DEX
        "Call DEX Wall":        round(call_delta_wall,    1),
        "Put DEX Wall":         round(put_delta_wall,     1),
        "DEX Flip":             round(dflip,              1),
        # VEX / Vanna
        "Vanna Flip":           round(vanna_flip,         1),
        # Scalars
        "Max Pain":             round(max_pain,           1),
        "Net GEX $B":           round(total_net_gex/1e9,  3),
        "Net DEX $B":           round(total_net_dex/1e9,  3),
        "Net VEX $M":           round(total_net_vex/1e6,  2),
        "Net CEX (daily)":      round(total_net_cex,       0),
        # Individual regimes
        "Regime":               regime,
        "DEX Bias":             dex_bias,
        "Vanna Regime":         vanna_regime,
        "Charm Flow":           charm_flow,
        # Combined engine
        "Intraday Condition":   intraday_condition,
        "Strategy":             strategy,
        "Condition Class":      condition_cls,
        "Vanna Note":           vanna_note,
        "Charm Note":           charm_note,
    }, filt

def expected_move(spot, net_b, d=20):
    # IV proxy: start at 18% for gold (lower than 30%), shrink slightly with positive GEX
    # Positive GEX = pinning regime = lower realised vol
    d   = max(int(d), 1)
    iv  = max(0.10, 0.18 - abs(net_b) * 0.01)
    em  = spot * iv * (d / 365.0) ** 0.5
    return round(spot - em, 1), round(spot + em, 1)

def backquant_block(label, lvl, spot, expiry):
    now  = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    d    = dte(expiry)
    lo, hi = expected_move(spot, lvl["Net GEX $B"], d)
    return (
        f"GEX + DEX + VANNA + CHARM Levels [{label}] - {now}\n"
        f"Regime: {lvl['Intraday Condition']}  |  {lvl['Strategy']}\n"
        f"Vanna: {lvl['Vanna Note']}  |  Charm: {lvl['Charm Note']}\n"
        f"---\n"
        f"Core GEX Levels:\n"
        f"HVL: ${lvl['GEX Flip']:,.1f}\n"
        f"Call Resistance: ${lvl['Call Wall']:,.1f}\n"
        f"Put Support: ${lvl['Put Wall']:,.1f}\n"
        f"Core DEX Levels:\n"
        f"Delta Flip: ${lvl['DEX Flip']:,.1f}\n"
        f"Call Delta Wall: ${lvl['Call DEX Wall']:,.1f}\n"
        f"Put Delta Wall: ${lvl['Put DEX Wall']:,.1f}\n"
        f"Vanna & Charm:\n"
        f"Vanna Flip: ${lvl['Vanna Flip']:,.1f}\n"
        f"Vanna Regime: {lvl['Vanna Regime']}\n"
        f"Charm Flow: {lvl['Charm Flow']}\n"
        f"Context:\n"
        f"Max Pain: ${lvl['Max Pain']:,.1f}\n"
        f"Expected Move: ${lo:,.1f} to ${hi:,.1f}\n"
    )

# ── CHART ─────────────────────────────────────────────────────────────────────
def gex_chart(filt, lvl, spot, label):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#080c10")
    ax.set_facecolor("#080c10")
    x  = filt["strike"]
    bw = (x.max()-x.min()) / max(len(x),1) * 0.75
    ax.bar(x, filt["call_gex"]/1e9, width=bw, color="#00e676", alpha=0.7, label="Call GEX")
    ax.bar(x, filt["put_gex"] /1e9, width=bw, color="#ff5252", alpha=0.7, label="Put GEX")
    ax.axhline(0, color="#2a3a2a", linewidth=0.8)
    for val, col, ls, lw, lbl in [
        (lvl["Call Wall"],  "#00e676", "-",  2.0, f"CALL WALL {lvl['Call Wall']}"),
        (lvl["Put Wall"],   "#ff5252", "-",  2.0, f"PUT WALL  {lvl['Put Wall']}"),
        (lvl["GEX Flip"],   "#00bcd4", "--", 1.5, f"GEX FLIP  {lvl['GEX Flip']}"),
        (lvl["Vanna Flip"], "#ce93d8", "-.", 1.5, f"VANNA FLIP {lvl['Vanna Flip']}"),
        (lvl["Max Pain"],   "#ffd740", ":",  1.2, f"MAX PAIN  {lvl['Max Pain']}"),
        (spot,              "#ffffff", "--", 1.0, f"SPOT      {spot:.1f}"),
    ]:
        ax.axvline(val, color=col, linestyle=ls, linewidth=lw, label=lbl)
    ax.set_title(
        f"{label}  |  {datetime.now().strftime('%d %b %Y')}  |  "
        f"Regime: {lvl['Regime']}  |  Net GEX ${lvl['Net GEX $B']}B",
        color="#8a9bb0", fontsize=9, pad=10)
    ax.set_xlabel("Strike", color="#4a6070", fontsize=8)
    ax.set_ylabel("GEX ($B)", color="#4a6070", fontsize=8)
    ax.tick_params(colors="#4a6070", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#1a2a1a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#0d1f2d", labelcolor="#c9d1d9",
              fontsize=7, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    return fig

# ── FETCH ALL DATA ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(_cache_buster: int = 0,
              xau_override: float = 0.0,
              us500_override: float = 0.0,
              us100_override: float = 0.0):
    results = {}

    # ── GOLD ──────────────────────────────────────────────────────────────────
    # XAU=X = interbank spot gold USD — closest to FXCM XAUUSD
    # GC=F  = CME front-month futures (yfinance sometimes returns wrong contract)
    # Cascade: XAU=X → GC=F → GLD ETF×scale
    gc_spot = None
    for gold_spot_sym in ["XAU=X", "GC=F"]:
        try:
            h = yf.Ticker(gold_spot_sym).history(period="5d")
            if not h.empty:
                p = float(h["Close"].dropna().iloc[-1])
                if p > 1500:   # sanity check — gold above $1500
                    gc_spot = p
                    break
        except:
            continue
    if gc_spot is None:
        # Last resort: GLD ETF × ~10.5 ratio
        try:
            gld_p = float(yf.Ticker("GLD").history(period="2d")["Close"].dropna().iloc[-1])
            gc_spot = round(gld_p * 10.5, 1)
        except:
            gc_spot = 3300.0  # hard fallback

    gold_source = None
    gold_raw    = None
    for days_back in range(0, 5):
        td = biz_day(days_back)
        df = _try_cme_json(td)
        if df is not None and len(df) > 10:
            gold_raw    = df
            gold_source = f"CME Direct — JSON ({td})"
            gc_expiry   = str(td)
            break
        time.sleep(0.5)
        df = _try_cme_html(td)
        if df is not None and len(df) > 10:
            gold_raw    = df
            gold_source = f"CME Direct — HTML ({td})"
            gc_expiry   = str(td)
            break
        time.sleep(0.5)

    if gold_raw is None:
        # gc_spot is now XAU=X (interbank spot) so proxy strikes scale correctly
        df_gld, gc_spot, gc_expiry, etf_spot, scale, proxy_sym = _gld_proxy(gc_spot, authoritative_spot=gc_spot)
        gold_raw    = df_gld
        gold_source = f"{proxy_sym} Proxy  (${etf_spot:.2f} × {scale:.2f})"
    else:
        # ── STRIKE CENTRING GUARD ─────────────────────────────────────────────
        # CME may return deferred-contract strikes (e.g. Aug/Dec 2026 ~$4,100–$4,500)
        # while gc_spot (XAU=X) is ~$3,320. Filter to ±25% of spot to guarantee
        # we're working with near-money options for the correct contract.
        lo, hi  = gc_spot * 0.75, gc_spot * 1.25
        centred = gold_raw[(gold_raw["strike"] >= lo) & (gold_raw["strike"] <= hi)]
        if len(centred) >= 10:
            gold_raw = centred.reset_index(drop=True)
        else:
            # Strikes are systematically off — wrong contract. Fall through to GLD proxy.
            st.warning(
                f"⚠️ CME strikes ({gold_raw['strike'].median():.0f}) are far from "
                f"spot ({gc_spot:.0f}). Using GLD proxy instead."
            )
            df_gld, gc_spot, gc_expiry, etf_spot, scale, proxy_sym = _gld_proxy(gc_spot, authoritative_spot=gc_spot)
            gold_raw    = df_gld
            gold_source = f"{proxy_sym} Proxy  (${etf_spot:.2f} × {scale:.2f})  [CME contract mismatch]"

    # If user provided a CFD override, re-scale strikes to that price space.
    # Greeks stay in native option space; only the display conversion changes.
    if xau_override > 1500:
        xau_display = xau_override
        cfd_scale = xau_display / gc_spot
        gold_raw_display = gold_raw.copy()
        gold_raw_display["strike"] = (gold_raw_display["strike"] * cfd_scale).round(1)
        _gold_dte = dte(gc_expiry)
        gold_df = enrich(gold_raw_display, xau_display, GC_MULTIPLIER, _gold_dte)
        gold_lvl, gold_filt = levels(gold_df, xau_display, "XAUUSD", days=_gold_dte)
        gold_block = backquant_block("XAUUSD", gold_lvl, xau_display, gc_expiry)
        gold_source += f"  |  ⚡ CFD override: {xau_display:.1f}"
    else:
        xau_display = gc_spot
        _gold_dte = dte(gc_expiry)
        gold_df = enrich(gold_raw, gc_spot, GC_MULTIPLIER, _gold_dte)
        gold_lvl, gold_filt = levels(gold_df, gc_spot, "XAUUSD", days=_gold_dte)
        gold_block = backquant_block("XAUUSD", gold_lvl, gc_spot, gc_expiry)

    results["gold"] = {
        "lvl": gold_lvl, "filt": gold_filt, "spot": xau_display,
        "auto_spot": gc_spot,
        "expiry": gc_expiry, "source": gold_source, "block": gold_block,
    }

    # ── SPY / US500 ───────────────────────────────────────────────────────────
    # Fetch SPX index spot first — this is the real US500 price (FXCM CFD tracks ^GSPC)
    try:
        spx = float(yf.Ticker("^GSPC").history(period="2d")["Close"].dropna().iloc[-1])
    except:
        spx = None

    spy   = yf.Ticker("SPY")
    s_spot= float(spy.history(period="2d")["Close"].dropna().iloc[-1])
    s_exp = spy.options[0]
    sch   = spy.option_chain(s_exp)
    s_df  = pd.merge(
        sch.calls[["strike","openInterest","impliedVolatility"]].rename(
            columns={"openInterest":"call_oi","impliedVolatility":"call_iv"}),
        sch.puts[["strike","openInterest","impliedVolatility"]].rename(
            columns={"openInterest":"put_oi","impliedVolatility":"put_iv"}),
        on="strike", how="outer").fillna(0)
    s_df["call_settle"] = 0.0; s_df["put_settle"] = 0.0

    if spx is None:
        spx = s_spot * 10.0
    # CFD override: user can correct auto-fetched ^GSPC if it's stale/wrong
    if us500_override > 1000:
        spx_display = us500_override
    else:
        spx_display = spx
    spx_scale = round(spx_display / s_spot, 4)

    # Scale SPY strikes to CFD display space before enriching — levels in CFD price space
    s_df_scaled = s_df.copy()
    s_df_scaled["strike"] = (s_df_scaled["strike"] * spx_scale).round(1)
    _s_dte   = dte(s_exp)
    s_rich   = enrich(s_df_scaled, spx_display, EQ_MULTIPLIER, _s_dte)
    s_lvl, s_filt = levels(s_rich, spx_display, "US500", days=_s_dte)
    s_block  = backquant_block("US500", s_lvl, spx_display, s_exp)

    results["spy"] = {
        "lvl": s_lvl, "filt": s_filt, "spot": s_spot, "index_spot": spx_display,
        "auto_index": spx,
        "expiry": s_exp, "scale": spx_scale, "block": s_block,
    }

    # ── QQQ / US100 ───────────────────────────────────────────────────────────
    # Fetch NDX index spot first — this is the real US100 price (FXCM CFD tracks ^NDX)
    try:
        ndx = float(yf.Ticker("^NDX").history(period="2d")["Close"].dropna().iloc[-1])
    except:
        ndx = None

    qqq   = yf.Ticker("QQQ")
    q_spot= float(qqq.history(period="2d")["Close"].dropna().iloc[-1])
    q_exp = qqq.options[0]
    qch   = qqq.option_chain(q_exp)
    q_df  = pd.merge(
        qch.calls[["strike","openInterest","impliedVolatility"]].rename(
            columns={"openInterest":"call_oi","impliedVolatility":"call_iv"}),
        qch.puts[["strike","openInterest","impliedVolatility"]].rename(
            columns={"openInterest":"put_oi","impliedVolatility":"put_iv"}),
        on="strike", how="outer").fillna(0)
    q_df["call_settle"] = 0.0; q_df["put_settle"] = 0.0

    if ndx is None:
        ndx = q_spot * 40.0
    # CFD override: user can correct auto-fetched ^NDX if it's stale/wrong
    if us100_override > 5000:
        ndx_display = us100_override
    else:
        ndx_display = ndx
    ndx_scale = round(ndx_display / q_spot, 4)

    # Scale QQQ strikes to CFD display space before enriching — levels in CFD price space
    q_df_scaled = q_df.copy()
    q_df_scaled["strike"] = (q_df_scaled["strike"] * ndx_scale).round(1)
    _q_dte   = dte(q_exp)
    q_rich   = enrich(q_df_scaled, ndx_display, EQ_MULTIPLIER, _q_dte)
    q_lvl, q_filt = levels(q_rich, ndx_display, "US100", days=_q_dte)
    q_block  = backquant_block("US100", q_lvl, ndx_display, q_exp)

    results["qqq"] = {
        "lvl": q_lvl, "filt": q_filt, "spot": q_spot, "index_spot": ndx_display,
        "auto_index": ndx,
        "expiry": q_exp, "scale": ndx_scale, "block": q_block,
    }

    return results

# ── LEVEL CARD RENDERER ───────────────────────────────────────────────────────
def render_level_card(lvl):
    reg_cls   = "regime-pos"    if "POSITIVE"   in lvl["Regime"]       else "regime-neg"
    dex_cls   = "regime-pos"    if "BULLISH"     in lvl["DEX Bias"]     else "regime-neg"
    van_cls   = "regime-purple" if "BULLISH"     in lvl["Vanna Regime"] else "regime-neg"
    chm_cls   = "regime-orange" if "BULLISH"     in lvl["Charm Flow"]   else "regime-neg"
    cond_cls  = lvl["Condition Class"]

    cex_val = lvl["Net CEX (daily)"]
    cex_str = f"+{cex_val:,.0f}" if cex_val >= 0 else f"{cex_val:,.0f}"

    st.markdown(f"""
    <div class="level-card">

      <div class="section-header">◈ INTRADAY REGIME ENGINE</div>
      <div class="level-row" style="padding:8px 0;">
        <span class="level-label">Condition</span>
        <span class="{cond_cls}" style="font-size:1rem;letter-spacing:0.5px;">{lvl['Intraday Condition']}</span>
      </div>
      <div class="level-row" style="padding:4px 0 8px 0;">
        <span class="level-label">Strategy</span>
        <span class="level-val" style="font-size:0.78rem;color:#8a9bb0;text-align:right;max-width:65%;">{lvl['Strategy']}</span>
      </div>
      <div class="level-row" style="padding:3px 0;">
        <span class="level-label">Vanna Signal</span>
        <span class="regime-purple" style="font-size:0.8rem;">{lvl['Vanna Note']}</span>
      </div>
      <div class="level-row" style="padding:3px 0 8px 0;">
        <span class="level-label">Charm Signal</span>
        <span class="regime-orange" style="font-size:0.8rem;">{lvl['Charm Note']}</span>
      </div>

      <div class="section-divider"></div>
      <div class="section-header">GEX</div>
      <div class="level-row"><span class="level-label">Regime</span>
        <span class="{reg_cls}">{lvl['Regime']}</span></div>
      <div class="level-row"><span class="level-label">GEX Flip / HVL</span>
        <span class="level-val lvl-cyan">{lvl['GEX Flip']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Call Wall</span>
        <span class="level-val lvl-green">{lvl['Call Wall']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Put Wall</span>
        <span class="level-val lvl-red">{lvl['Put Wall']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Net GEX</span>
        <span class="level-val">${lvl['Net GEX $B']}B</span></div>

      <div class="section-divider"></div>
      <div class="section-header">DEX</div>
      <div class="level-row"><span class="level-label">DEX Bias</span>
        <span class="{dex_cls}">{lvl['DEX Bias']}</span></div>
      <div class="level-row"><span class="level-label">Delta Flip</span>
        <span class="level-val lvl-cyan">{lvl['DEX Flip']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Call Delta Wall</span>
        <span class="level-val lvl-green">{lvl['Call DEX Wall']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Put Delta Wall</span>
        <span class="level-val lvl-red">{lvl['Put DEX Wall']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Net DEX</span>
        <span class="level-val">${lvl['Net DEX $B']}B</span></div>

      <div class="section-divider"></div>
      <div class="section-header">VANNA & CHARM</div>
      <div class="level-row"><span class="level-label">Vanna Regime</span>
        <span class="{van_cls}">{lvl['Vanna Regime']}</span></div>
      <div class="level-row"><span class="level-label">Vanna Flip</span>
        <span class="level-val lvl-purple">{lvl['Vanna Flip']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">Net VEX</span>
        <span class="level-val lvl-purple">${lvl['Net VEX $M']}M</span></div>
      <div class="level-row"><span class="level-label">Charm Flow</span>
        <span class="{chm_cls}">{lvl['Charm Flow']}</span></div>
      <div class="level-row"><span class="level-label">Net CEX / day</span>
        <span class="level-val lvl-orange">{cex_str} Δ</span></div>

      <div class="section-divider"></div>
      <div class="section-header">CONTEXT</div>
      <div class="level-row"><span class="level-label">Max Pain</span>
        <span class="level-val lvl-yellow">{lvl['Max Pain']:,.1f}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gex-header">
  <h1>GEX ENGINE</h1>
  <p>XAUUSD · US500 · US100 &nbsp;|&nbsp; GEX · DEX · Vanna · Charm &nbsp;|&nbsp; BackQuant Format</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#0d1117;border:1px solid #1a2a3a;border-radius:8px;padding:10px 16px 6px 16px;
            font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a6070;margin-bottom:12px;">
  ⚡ CFD SPOT OVERRIDES &nbsp;—&nbsp; leave at 0 to use auto-fetched prices (XAU=X · ^GSPC · ^NDX)
</div>
""", unsafe_allow_html=True)

ov_col1, ov_col2, ov_col3 = st.columns(3)
with ov_col1:
    xau_override = st.number_input(
        "XAUUSD CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Override auto-fetched XAU=X. Set to 0 to use auto price.")
with ov_col2:
    us500_override = st.number_input(
        "US500 CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Override auto-fetched ^GSPC. Set to 0 to use auto price.")
with ov_col3:
    us100_override = st.number_input(
        "US100 CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Override auto-fetched ^NDX. Set to 0 to use auto price.")

btn_col, cc_col = st.columns([4, 1])
with btn_col:
    run = st.button("⚡ Generate Levels")
with cc_col:
    if st.button("🗑 Cache"):
        fetch_all.clear()
        st.success("Cache cleared")
        st.rerun()

# ── RESULTS ───────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching live spots and computing GEX + Vanna + Charm levels…"):
        try:
            data = fetch_all(
                xau_override=xau_override,
                us500_override=us500_override,
                us100_override=us100_override,
            )
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    st.success(f"Levels ready — {datetime.now().strftime('%d %b %Y  %H:%M')}")

    tab1, tab2, tab3 = st.tabs(["🥇 XAUUSD", "📈 US500", "💻 US100"])

    with tab1:
        g = data["gold"]
        badge_cls = "badge-cme" if "CME" in g["source"] else "badge-gld"
        # Show override indicator if active
        spot_note = f"  |  auto: {g['auto_spot']:.1f}" if xau_override > 1500 else f"  auto: {g['auto_spot']:.1f}"
        st.markdown(f'<span class="source-badge {badge_cls}">SOURCE: {g["source"]}{spot_note}</span>',
                    unsafe_allow_html=True)
        render_level_card(g["lvl"])
        st.markdown("**📋 BackQuant Paste Block — copy and paste into indicator settings:**")
        st.code(g["block"], language=None)
        # Chart spot = CFD display spot (already correct after override applied)
        st.pyplot(gex_chart(g["filt"], g["lvl"], g["spot"], "XAUUSD"))

    with tab2:
        s = data["spy"]
        override_tag = f"  ⚡ override" if us500_override > 1000 else ""
        auto_note = f"  auto: {s['auto_index']:.1f}" if us500_override > 1000 else ""
        st.markdown(f'<span class="source-badge badge-cme">US500 {s["index_spot"]:.1f}{override_tag}  (SPY×{s["scale"]:.4f}){auto_note}</span>',
                    unsafe_allow_html=True)
        render_level_card(s["lvl"])
        st.markdown("**📋 BackQuant Paste Block:**")
        st.code(s["block"], language=None)
        # Chart: pass index_spot so SPOT line aligns with CFD levels (not ETF price)
        st.pyplot(gex_chart(s["filt"], s["lvl"], s["index_spot"], "US500"))

    with tab3:
        q = data["qqq"]
        override_tag = f"  ⚡ override" if us100_override > 5000 else ""
        auto_note = f"  auto: {q['auto_index']:.1f}" if us100_override > 5000 else ""
        st.markdown(f'<span class="source-badge badge-cme">US100 {q["index_spot"]:.1f}{override_tag}  (QQQ×{q["scale"]:.4f}){auto_note}</span>',
                    unsafe_allow_html=True)
        render_level_card(q["lvl"])
        st.markdown("**📋 BackQuant Paste Block:**")
        st.code(q["block"], language=None)
        # Chart: pass index_spot so SPOT line aligns with CFD levels (not ETF price)
        st.pyplot(gex_chart(q["filt"], q["lvl"], q["index_spot"], "US100"))

    st.markdown("""
    <div class="footer">
      Levels cached for 6 hours · CME Direct → GLD Proxy fallback · GC=F removed<br>
      Data is indicative — always verify with your broker before trading
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #3d5268;">
      <div style="font-size: 3rem; margin-bottom: 16px;">📊</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
        Enter your XAUUSD spot price above<br>then click Generate Levels
      </div>
    </div>
    """, unsafe_allow_html=True)
