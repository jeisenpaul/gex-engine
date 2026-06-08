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
                        sv = lambda x: float(str(x).replace(",","").replace("-","0") or 0) if True else 0.0
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

def _gld_proxy(gc_spot):
    # Try multiple gold ETF proxies in order: GLD → IAU → SGOL
    GOLD_PROXIES = ["GLD", "IAU", "SGOL"]
    for ticker_sym in GOLD_PROXIES:
        try:
            t        = yf.Ticker(ticker_sym)
            hist     = t.history(period="5d")
            if hist.empty: continue
            etf_spot = float(hist["Close"].dropna().iloc[-1])
            scale    = gc_spot / etf_spot
            exps     = t.options
            if not exps: continue
            best_expiry, best_oi, best_calls, best_puts = None, 0, None, None
            # Scan up to 6 expirations for richest OI
            for exp in exps[:6]:
                try:
                    ch    = t.option_chain(exp)
                    total = ch.calls["openInterest"].sum() + ch.puts["openInterest"].sum()
                    if total > best_oi:
                        best_oi, best_expiry = total, exp
                        best_calls, best_puts = ch.calls, ch.puts
                except: continue
            if best_calls is None or best_oi < 100:
                continue  # try next proxy
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
            return df, gc_spot, best_expiry, etf_spot, scale, ticker_sym
        except: continue
    raise RuntimeError("All gold proxies failed (GLD, IAU, SGOL). Try again in a few minutes.")

# ── CORE COMPUTATION ──────────────────────────────────────────────────────────
def enrich(df, spot, multiplier, days=30):
    T  = max(days / 365.0, 1/365)
    df = df.copy()
    civ_l, piv_l, cg_l, pg_l, cd_l, pd_l = [], [], [], [], [], []
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
    df["call_iv_c"] = civ_l; df["put_iv_c"] = piv_l
    df["call_gamma"]= cg_l;  df["put_gamma"] = pg_l
    df["call_delta"]= cd_l;  df["put_delta"] = pd_l
    df["call_gex"]  =  df["call_gamma"] * df["call_oi"] * multiplier * spot**2
    df["put_gex"]   = -df["put_gamma"]  * df["put_oi"]  * multiplier * spot**2
    df["net_gex"]   =  df["call_gex"]   + df["put_gex"]
    df["call_dex"]  =  df["call_delta"] * df["call_oi"] * multiplier * spot
    df["put_dex"]   =  df["put_delta"]  * df["put_oi"]  * multiplier * spot
    df["net_dex"]   =  df["call_dex"]   + df["put_dex"]
    return df

def levels(df, spot, label=""):
    mask = (df["strike"] >= spot*0.75) & (df["strike"] <= spot*1.25)
    filt = df[mask].sort_values("strike").reset_index(drop=True)
    if filt.empty:
        filt = df.sort_values("strike").reset_index(drop=True)
    call_wall = float(filt.loc[filt["call_gex"].idxmax(), "strike"])
    put_wall  = float(filt.loc[filt["put_gex"].idxmin(),  "strike"])
    call_delta_wall = float(filt.loc[filt["call_dex"].idxmax(), "strike"])
    put_delta_wall  = float(filt.loc[filt["put_dex"].idxmin(),  "strike"])
    cum       = filt["net_gex"].cumsum().values
    crossings = np.where(np.diff(np.sign(cum)))[0]
    if len(crossings) > 0:
        ci  = crossings[len(crossings)//2]
        ci1 = min(ci+1, len(filt)-1)
        s0, s1 = filt.iloc[ci]["strike"], filt.iloc[ci1]["strike"]
        v0, v1 = float(cum[ci]), float(cum[ci1])
        gflip = (s0 - v0*(s1-s0)/(v1-v0)) if (v1-v0) != 0 else s0
    else:
        gflip = float(filt.iloc[np.argmin(np.abs(cum))]["strike"])
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
    sa  = df["strike"].values
    coi = df["call_oi"].values.astype(float)
    poi = df["put_oi"].values.astype(float)
    lss = [np.sum(np.maximum(sa-s,0)*coi)+np.sum(np.maximum(s-sa,0)*poi) for s in sa]
    max_pain  = float(sa[np.argmin(lss)])
    total_net = float(filt["net_gex"].sum())
    total_dex = float(filt["net_dex"].sum())
    regime    = "POSITIVE ▲" if spot > gflip else "NEGATIVE ▼"
    dex_bias  = "BULLISH ▲" if total_dex > 0 else "BEARISH ▼"
    return {
        "Call Wall":  round(call_wall, 1),
        "Put Wall":   round(put_wall,  1),
        "GEX Flip":   round(gflip,     1),
        "Call DEX Wall": round(call_delta_wall, 1),
        "Put DEX Wall":  round(put_delta_wall,  1),
        "DEX Flip":      round(dflip,           1),
        "Max Pain":   round(max_pain,  1),
        "Net GEX $B": round(total_net/1e9, 3),
        "Net DEX $B": round(total_dex/1e9, 3),
        "Regime":     regime,
        "DEX Bias":   dex_bias,
    }, filt

def expected_move(spot, net_b, d=20):
    iv = max(0.08, 0.30 - abs(net_b)*0.02)
    em = spot * iv * max(d,1)**0.5 / 365**0.5
    return round(spot-em,1), round(spot+em,1)

def backquant_block(label, lvl, spot, expiry):
    now  = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    d    = dte(expiry)
    lo, hi = expected_move(spot, lvl["Net GEX $B"], d)
    return (
        f"GEX + DEX Levels [{label}] - {now}\n"
        f"Core GEX Levels:\n"
        f"HVL: ${lvl['GEX Flip']:,.1f}\n"
        f"Call Resistance: ${lvl['Call Wall']:,.1f}\n"
        f"Put Support: ${lvl['Put Wall']:,.1f}\n"
        f"Core DEX Levels:\n"
        f"Delta Flip: ${lvl['DEX Flip']:,.1f}\n"
        f"Call Delta Wall: ${lvl['Call DEX Wall']:,.1f}\n"
        f"Put Delta Wall: ${lvl['Put DEX Wall']:,.1f}\n"
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
        (lvl["Call Wall"], "#00e676", "-",  2.0, f"CALL WALL {lvl['Call Wall']}"),
        (lvl["Put Wall"],  "#ff5252", "-",  2.0, f"PUT WALL  {lvl['Put Wall']}"),
        (lvl["GEX Flip"],  "#00bcd4", "--", 1.5, f"GEX FLIP  {lvl['GEX Flip']}"),
        (lvl["Max Pain"],  "#ffd740", ":",  1.2, f"MAX PAIN  {lvl['Max Pain']}"),
        (spot,             "#ffffff", "--", 1.0, f"SPOT      {spot:.1f}"),
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
@st.cache_data(ttl=21600, show_spinner=False)
def fetch_all(xauusd_spot: float):
    results = {}

    # ── GOLD ──────────────────────────────────────────────────────────────────
    try:
        gc_spot = float(yf.Ticker("GC=F").history(period="5d")["Close"].dropna().iloc[-1])
    except:
        gc_spot = xauusd_spot

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
        df_gld, gc_spot, gc_expiry, etf_spot, scale, proxy_sym = _gld_proxy(gc_spot)
        gold_raw    = df_gld
        gold_source = f"{proxy_sym} Proxy  (${etf_spot:.2f} 00d7 {scale:.2f})"

    gold_df = enrich(gold_raw, gc_spot, GC_MULTIPLIER, dte(gc_expiry))
    gold_lvl, gold_filt = levels(gold_df, gc_spot, "XAUUSD")
    gold_block = backquant_block("XAUUSD", gold_lvl, gc_spot, gc_expiry)

    results["gold"] = {
        "lvl": gold_lvl, "filt": gold_filt, "spot": gc_spot,
        "expiry": gc_expiry, "source": gold_source, "block": gold_block,
    }

    # ── SPY / US500 ───────────────────────────────────────────────────────────
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
    s_rich   = enrich(s_df, s_spot, EQ_MULTIPLIER, dte(s_exp))
    s_lvl, s_filt = levels(s_rich, s_spot, "US500")
    try:
        spx = float(yf.Ticker("^GSPC").history(period="2d")["Close"].dropna().iloc[-1])
    except:
        spx = s_spot * 10.0
    spx_scale = round(spx / s_spot, 4)
    s_lvl_sc  = {k: round(v*spx_scale,1) if k in ["Call Wall","Put Wall","GEX Flip","Call DEX Wall","Put DEX Wall","DEX Flip","Max Pain"] else v
                 for k,v in s_lvl.items()}
    s_block   = backquant_block("US500-SPX", s_lvl_sc, spx, s_exp)

    results["spy"] = {
        "lvl": s_lvl_sc, "filt": s_filt, "spot": s_spot, "index_spot": spx,
        "expiry": s_exp, "scale": spx_scale, "block": s_block,
    }

    # ── QQQ / US100 ───────────────────────────────────────────────────────────
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
    q_rich   = enrich(q_df, q_spot, EQ_MULTIPLIER, dte(q_exp))
    q_lvl, q_filt = levels(q_rich, q_spot, "US100")
    try:
        ndx = float(yf.Ticker("^NDX").history(period="2d")["Close"].dropna().iloc[-1])
    except:
        ndx = q_spot * 40.0
    ndx_scale = round(ndx / q_spot, 4)
    q_lvl_sc  = {k: round(v*ndx_scale,1) if k in ["Call Wall","Put Wall","GEX Flip","Call DEX Wall","Put DEX Wall","DEX Flip","Max Pain"] else v
                 for k,v in q_lvl.items()}
    q_block   = backquant_block("US100-NDX", q_lvl_sc, ndx, q_exp)

    results["qqq"] = {
        "lvl": q_lvl_sc, "filt": q_filt, "spot": q_spot, "index_spot": ndx,
        "expiry": q_exp, "scale": ndx_scale, "block": q_block,
    }

    return results

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gex-header">
  <h1>GEX ENGINE</h1>
  <p>XAUUSD · US500 · US100 &nbsp;|&nbsp; CME Direct → GLD Proxy &nbsp;|&nbsp; BackQuant Format</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    xauusd_input = st.number_input(
        "XAUUSD Spot Price",
        min_value=1000.0, max_value=15000.0,
        value=3300.0, step=0.1, format="%.1f",
        help="Enter current gold price from your broker"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡ Generate Levels")

# ── RESULTS ───────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Fetching data and computing GEX levels…"):
        try:
            data = fetch_all(xauusd_input)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    st.success(f"Levels ready — {datetime.now().strftime('%d %b %Y  %H:%M')}")

    tab1, tab2, tab3 = st.tabs(["🥇 XAUUSD", "📈 US500", "💻 US100"])

    # ── XAUUSD ────────────────────────────────────────────────────────────────
    with tab1:
        g = data["gold"]
        badge_cls = "badge-cme" if "CME" in g["source"] else "badge-gld"
        st.markdown(f'<span class="source-badge {badge_cls}">SOURCE: {g["source"]}</span>',
                    unsafe_allow_html=True)
        lvl = g["lvl"]
        reg_cls = "regime-pos" if "POSITIVE" in lvl["Regime"] else "regime-neg"
        dex_cls = "regime-pos" if "BULLISH" in lvl["DEX Bias"] else "regime-neg"
        st.markdown(f"""
        <div class="level-card">
          <div class="level-row"><span class="level-label">Regime</span>
            <span class="{reg_cls}">{lvl['Regime']}</span></div>
          <div class="level-row"><span class="level-label">Call Wall</span>
            <span class="level-val lvl-green">{lvl['Call Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Put Wall</span>
            <span class="level-val lvl-red">{lvl['Put Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">GEX Flip / HVL</span>
            <span class="level-val lvl-cyan">{lvl['GEX Flip']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">DEX Bias</span>
            <span class="{dex_cls}">{lvl['DEX Bias']}</span></div>
          <div class="level-row"><span class="level-label">Delta Flip</span>
            <span class="level-val lvl-cyan">{lvl['DEX Flip']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Call Delta Wall</span>
            <span class="level-val lvl-green">{lvl['Call DEX Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Put Delta Wall</span>
            <span class="level-val lvl-red">{lvl['Put DEX Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Max Pain</span>
            <span class="level-val lvl-yellow">{lvl['Max Pain']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Net GEX</span>
            <span class="level-val">${lvl['Net GEX $B']}B</span></div>
          <div class="level-row"><span class="level-label">Net DEX</span>
            <span class="level-val">${lvl['Net DEX $B']}B</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**📋 BackQuant Paste Block — copy and paste into indicator settings:**")
        st.code(g["block"], language=None)
        st.pyplot(gex_chart(g["filt"], lvl, g["spot"], "XAUUSD"))

    # ── US500 ─────────────────────────────────────────────────────────────────
    with tab2:
        s = data["spy"]
        lvl = s["lvl"]
        reg_cls = "regime-pos" if "POSITIVE" in lvl["Regime"] else "regime-neg"
        dex_cls = "regime-pos" if "BULLISH" in lvl["DEX Bias"] else "regime-neg"
        st.markdown(f'<span class="source-badge badge-cme">SPY {s["spot"]:.2f} × {s["scale"]:.2f} = SPX {s["index_spot"]:.0f}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="level-card">
          <div class="level-row"><span class="level-label">Regime</span>
            <span class="{reg_cls}">{lvl['Regime']}</span></div>
          <div class="level-row"><span class="level-label">Call Wall</span>
            <span class="level-val lvl-green">{lvl['Call Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Put Wall</span>
            <span class="level-val lvl-red">{lvl['Put Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">GEX Flip / HVL</span>
            <span class="level-val lvl-cyan">{lvl['GEX Flip']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">DEX Bias</span>
            <span class="{dex_cls}">{lvl['DEX Bias']}</span></div>
          <div class="level-row"><span class="level-label">Delta Flip</span>
            <span class="level-val lvl-cyan">{lvl['DEX Flip']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Call Delta Wall</span>
            <span class="level-val lvl-green">{lvl['Call DEX Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Put Delta Wall</span>
            <span class="level-val lvl-red">{lvl['Put DEX Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Max Pain</span>
            <span class="level-val lvl-yellow">{lvl['Max Pain']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Net GEX</span>
            <span class="level-val">${lvl['Net GEX $B']}B</span></div>
          <div class="level-row"><span class="level-label">Net DEX</span>
            <span class="level-val">${lvl['Net DEX $B']}B</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**📋 BackQuant Paste Block:**")
        st.code(s["block"], language=None)
        st.pyplot(gex_chart(s["filt"], s["lvl"], s["spot"], "US500 (SPY)"))

    # ── US100 ─────────────────────────────────────────────────────────────────
    with tab3:
        q = data["qqq"]
        lvl = q["lvl"]
        reg_cls = "regime-pos" if "POSITIVE" in lvl["Regime"] else "regime-neg"
        dex_cls = "regime-pos" if "BULLISH" in lvl["DEX Bias"] else "regime-neg"
        st.markdown(f'<span class="source-badge badge-cme">QQQ {q["spot"]:.2f} × {q["scale"]:.2f} = NDX {q["index_spot"]:.0f}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="level-card">
          <div class="level-row"><span class="level-label">Regime</span>
            <span class="{reg_cls}">{lvl['Regime']}</span></div>
          <div class="level-row"><span class="level-label">Call Wall</span>
            <span class="level-val lvl-green">{lvl['Call Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Put Wall</span>
            <span class="level-val lvl-red">{lvl['Put Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">GEX Flip / HVL</span>
            <span class="level-val lvl-cyan">{lvl['GEX Flip']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">DEX Bias</span>
            <span class="{dex_cls}">{lvl['DEX Bias']}</span></div>
          <div class="level-row"><span class="level-label">Delta Flip</span>
            <span class="level-val lvl-cyan">{lvl['DEX Flip']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Call Delta Wall</span>
            <span class="level-val lvl-green">{lvl['Call DEX Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Put Delta Wall</span>
            <span class="level-val lvl-red">{lvl['Put DEX Wall']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Max Pain</span>
            <span class="level-val lvl-yellow">{lvl['Max Pain']:,.1f}</span></div>
          <div class="level-row"><span class="level-label">Net GEX</span>
            <span class="level-val">${lvl['Net GEX $B']}B</span></div>
          <div class="level-row"><span class="level-label">Net DEX</span>
            <span class="level-val">${lvl['Net DEX $B']}B</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**📋 BackQuant Paste Block:**")
        st.code(q["block"], language=None)
        st.pyplot(gex_chart(q["filt"], q["lvl"], q["spot"], "US100 (QQQ)"))

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
