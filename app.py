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
    overflow: hidden;
}
.level-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    padding: 5px 0;
    border-bottom: 1px solid #15253a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.level-row:last-child { border-bottom: none; }
.level-label {
    color: #8a9bb0;
    flex: 0 0 42%;
    min-width: 0;
    overflow-wrap: anywhere;
}
.level-val {
    color: #e8eaed;
    font-weight: 600;
}
.level-row span:last-child {
    font-weight: 600;
    flex: 1 1 auto;
    min-width: 0;
    text-align: right;
    white-space: normal;
    overflow-wrap: anywhere;
}
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
@media (max-width: 520px) {
    .level-card { padding: 12px 14px; }
    .level-row {
        display: grid;
        grid-template-columns: minmax(92px, 38%) minmax(0, 1fr);
        gap: 10px;
    }
    .level-label,
    .level-val,
    .level-row span:last-child {
        flex-basis: auto;
    }
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

def get_atm_iv(df, spot):
    """
    Extract ATM implied volatility — the strike closest to spot.
    This is the correct IV input for Vanna and regime logic.
    Chain-average IV is distorted by deep OTM strikes; ATM IV reflects
    what dealers are actually pricing for near-money hedging.
    Returns the average of ATM call_iv and put_iv (put-call ATM average).
    """
    if df is None or df.empty:
        return 0.20
    atm_idx = (df["strike"] - spot).abs().idxmin()
    civ = float(df.loc[atm_idx, "call_iv"]) if "call_iv" in df.columns else 0.0
    piv = float(df.loc[atm_idx, "put_iv"])  if "put_iv"  in df.columns else 0.0
    # Fall back to enriched columns if raw IV is zero
    if civ <= 0.01 and "call_iv_c" in df.columns:
        civ = float(df.loc[atm_idx, "call_iv_c"])
    if piv <= 0.01 and "put_iv_c" in df.columns:
        piv = float(df.loc[atm_idx, "put_iv_c"])
    atm = (civ + piv) / 2 if (civ > 0.01 and piv > 0.01) else max(civ, piv)
    return atm if atm > 0.01 else 0.20

def store_session_open_iv(key, current_iv):
    """
    Store IV at session open — NOT first refresh IV.
    Uses NY market open time (09:30 ET) as the reference point.
    If it is before market open or a new trading day, resets the stored IV.
    Returns (open_iv, iv_change, iv_change_pct).
    """
    import pytz
    ny_tz    = pytz.timezone("America/New_York")
    now_ny   = datetime.now(ny_tz)
    today_str = now_ny.strftime("%Y-%m-%d")
    open_key  = f"{key}_open_iv"
    date_key  = f"{key}_open_date"
    hour_ny   = now_ny.hour + now_ny.minute / 60

    stored_date = st.session_state.get(date_key, "")
    stored_iv   = st.session_state.get(open_key, None)

    # Reset if: new trading day OR before 09:30 (pre-market — don't anchor to pre-market IV)
    is_new_day    = stored_date != today_str
    is_premarket  = hour_ny < 9.5
    is_afterhours = hour_ny >= 17.0

    if is_new_day or stored_iv is None:
        if not is_premarket:
            # Market is open — anchor to this IV as today's open reference
            st.session_state[open_key]  = current_iv
            st.session_state[date_key]  = today_str
            stored_iv = current_iv
        else:
            # Pre-market — don't store yet, no reliable anchor
            stored_iv = current_iv

    open_iv        = stored_iv
    iv_change      = current_iv - open_iv
    iv_change_pct  = (iv_change / open_iv * 100) if open_iv > 0 else 0.0
    return open_iv, iv_change, iv_change_pct


def implied_vol(price, S, K, T, r, opt):
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

def _yf_history_with_retry(ticker_sym, period="5d", retries=3, backoff=1.5):
    """
    yfinance wrapper with retry + exponential backoff.
    Yahoo Finance rate-limits aggressively — a single timeout shouldn't kill the app.
    """
    import time
    for attempt in range(retries):
        try:
            t    = yf.Ticker(ticker_sym)
            hist = t.history(period=period)
            if not hist.empty:
                return t, hist
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    return None, pd.DataFrame()


def _yf_option_chain_with_retry(ticker, exp, retries=3, backoff=1.5):
    """Fetch a single option chain expiry with retry."""
    import time
    for attempt in range(retries):
        try:
            ch = ticker.option_chain(exp)
            if ch.calls is not None and len(ch.calls) > 0:
                return ch
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    return None


def _gld_proxy(gc_spot, authoritative_spot=None):
    """
    Fetch gold options via GLD / IAU / SGOL ETF proxy.
    Priority: GLD → IAU → SGOL → session cache → synthetic fallback (never crashes).

    Notes:
    - GC=F removed: yfinance GC=F options uses CME futures chain structure
      (different columns, no impliedVolatility field) — always fails silently.
    - Session cache saves last good dataset; used when all live fetches fail.
    - Synthetic fallback generates a minimal ATM-centred chain from gc_spot
      so the app always produces output, clearly labelled as estimated.
    """
    GOLD_PROXIES = ["GLD", "IAU", "SGOL"]

    for ticker_sym in GOLD_PROXIES:
        try:
            t, hist = _yf_history_with_retry(ticker_sym, period="5d")
            if t is None or hist.empty:
                continue
            etf_spot = float(hist["Close"].dropna().iloc[-1])
            if etf_spot <= 0:
                continue
            scale = (authoritative_spot if authoritative_spot else gc_spot) / etf_spot

            # Fetch option expiries with retry
            exps = None
            for _ in range(3):
                try:
                    exps = t.options
                    if exps:
                        break
                except Exception:
                    import time; time.sleep(1.0)
            if not exps:
                continue

            # Scan first 3 expiries only — keeps HTTP calls low
            best_expiry, best_oi, best_calls, best_puts = None, 0, None, None
            for exp in exps[:3]:
                ch = _yf_option_chain_with_retry(t, exp)
                if ch is None:
                    continue
                if ch.calls is None or ch.puts is None:
                    continue
                if ch.calls.empty or ch.puts.empty:
                    continue
                total = int(ch.calls["openInterest"].sum()) + int(ch.puts["openInterest"].sum())
                if total > best_oi:
                    best_oi, best_expiry = total, exp
                    best_calls, best_puts = ch.calls, ch.puts

            if best_calls is None or best_oi < 50:
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

            # Save to session cache — used as fallback if future fetches fail
            st.session_state["gold_proxy_cache"] = {
                "df": df.copy(), "spot": spot_out, "expiry": best_expiry,
                "etf_spot": etf_spot, "scale": scale, "sym": ticker_sym,
            }
            return df, spot_out, best_expiry, etf_spot, scale, ticker_sym

        except Exception:
            continue

    # ── All live fetches failed — try session cache ───────────────────────────
    cached = st.session_state.get("gold_proxy_cache")
    if cached is not None:
        st.warning(
            "⚠️ Gold proxy temporarily unavailable (GLD / IAU / SGOL). "
            "Using last successful dataset — levels may be slightly stale.",
        )
        return (
            cached["df"], cached["spot"], cached["expiry"],
            cached["etf_spot"], cached["scale"], cached["sym"] + " [cached]"
        )

    # ── No cache either — synthetic fallback, never crash ────────────────────
    # Build a minimal ATM-centred chain from gc_spot so the app stays alive.
    # Strikes spaced at $10 intervals, ±15% from spot, flat 20% IV assumption.
    st.warning(
        "⚠️ Gold data temporarily unavailable (Yahoo Finance rate limit). "
        "Showing estimated levels based on spot price only. "
        "Refresh in 60 seconds for live data.",
    )
    spot_ref  = authoritative_spot if authoritative_spot else gc_spot
    strikes   = [round(spot_ref * (1 + i * 0.01), 1) for i in range(-15, 16)]
    syn_df    = pd.DataFrame({
        "strike":       strikes,
        "call_oi":      [max(100, int(1000 * (1 - abs(i)*0.06))) for i in range(-15, 16)],
        "put_oi":       [max(100, int(1000 * (1 - abs(i)*0.06))) for i in range(-15, 16)],
        "call_iv":      [0.20] * 31,
        "put_iv":       [0.20] * 31,
        "call_settle":  [0.0]  * 31,
        "put_settle":   [0.0]  * 31,
    })
    import datetime as _dt
    syn_expiry = (_dt.date.today() + _dt.timedelta(days=7)).strftime("%Y-%m-%d")
    return syn_df, spot_ref, syn_expiry, spot_ref / 10.0, 10.0, "SYNTHETIC [live data unavailable]"

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

def levels(df, spot, label="", days=30, iv_change=0.0, price_change=0.0):
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
    gex_positive = spot > gflip
    regime       = "POSITIVE ▲" if gex_positive else "NEGATIVE ▼"

    dex_positive = total_net_dex > 0
    dex_bias     = "BULLISH ▲" if dex_positive else "BEARISH ▼"

    # ── Vanna regime — price × IV direction (document Issue 4) ────────────────
    # The document explicitly states:
    #   Price ↑ + IV ↓ = true bullish Vanna (squeeze)
    #   Price ↓ + IV ↑ = true bearish Vanna (crash acceleration)
    # We also need net VEX sign to know which direction IV pressure moves dealers.
    # Combined: price direction + iv direction + net VEX sign.
    vex_positive   = total_net_vex > 0
    price_up       = price_change >= 0
    iv_falling     = iv_change <= 0

    # True bullish Vanna squeeze: price up AND IV falling
    # True bearish Vanna:         price down AND IV rising
    # Neutral / mixed:            price and IV moving same direction
    if price_up and iv_falling:
        # Classic squeeze condition — confirm with VEX
        if not vex_positive:
            # VEX < 0: dealers net short vanna → IV drop forces them to BUY spot → amplifies squeeze
            vanna_regime = "VOL-CRUSH BULLISH ▲"
            vanna_note   = "⚡ PRICE↑ + IV↓ + VEX− = SQUEEZE FUEL"
        else:
            # VEX > 0: dealers net long vanna → IV drop forces them to SELL → squeeze is fading
            vanna_regime = "VOL-CRUSH BEARISH ▼"
            vanna_note   = "⚠ PRICE↑ + IV↓ + VEX+ = RALLY FADING"
    elif not price_up and not iv_falling:
        # Classic crash acceleration — confirm with VEX
        if vex_positive:
            # VEX > 0: dealers net long vanna → IV spike forces them to BUY → provides floor
            vanna_regime = "VOL-SPIKE BULLISH ▲"
            vanna_note   = "⚠ PRICE↓ + IV↑ + VEX+ = FLOOR POSSIBLE"
        else:
            # VEX < 0: dealers net short vanna → IV spike forces them to SELL → crash accelerator
            vanna_regime = "VOL-SPIKE BEARISH ▼"
            vanna_note   = "⚡ PRICE↓ + IV↑ + VEX− = CRASH ACCELERATOR"
    elif price_up and not iv_falling:
        # Price up but IV also rising — not a clean squeeze, unusual
        vanna_regime = "VOL-SPIKE BEARISH ▼"
        vanna_note   = f"⚠ PRICE↑ + IV↑ — UNSTABLE · WATCH REVERSAL"
    else:
        # Price down but IV also falling — dead range or exhaustion
        vanna_regime = "VOL-CRUSH BULLISH ▲"
        vanna_note   = f"⏱ PRICE↓ + IV↓ — RANGE / EXHAUSTION"

    # ── Charm regime — DTE-weighted ───────────────────────────────────────────
    cex_positive   = total_net_cex > 0
    charm_flow     = "BULLISH BLEED ▲" if cex_positive else "BEARISH BLEED ▼"
    charm_reliable = days < 21

    if charm_reliable:
        charm_note = "📈 AFTERNOON MELT-UP BIAS" if cex_positive else "📉 AFTERNOON FADE BIAS"
    else:
        charm_note = f"⏱ CHARM: {'↑' if cex_positive else '↓'} (low weight — {days}+ DTE)"

    # ── Combined Regime Engine — four quadrants ────────────────────────────────
    # Primary: GEX + DEX quadrant
    # Modified by: Vanna (acceleration/fade) + Charm (intraday time direction)
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
    else:
        intraday_condition = "CRASH / TREND"
        strategy           = "Momentum shorts · Avoid catching bottoms · Trail stops"
        condition_cls      = "regime-neg"

    # ── Execution triggers (document Issue 5) ─────────────────────────────────
    # These are conditional entry filters — not signals by themselves.
    # They tell you WHEN the regime condition is actionable.
    if gex_positive and dex_positive and price_up and iv_falling:
        entry_trigger = "✅ LONG: regime confirmed · reclaim GEX flip · IV falling"
        trigger_cls   = "regime-pos"
    elif gex_positive and not dex_positive and not price_up and not iv_falling:
        entry_trigger = "✅ SHORT: regime confirmed · below GEX flip · IV rising"
        trigger_cls   = "regime-neg"
    elif not gex_positive and dex_positive and price_up and iv_falling:
        entry_trigger = "⚡ SQUEEZE ENTRY: breakout above GEX flip · momentum long"
        trigger_cls   = "regime-orange"
    elif not gex_positive and not dex_positive and not price_up and not iv_falling:
        entry_trigger = "⚡ TREND SHORT: breakdown · IV rising · avoid longs"
        trigger_cls   = "regime-neg"
    elif price_change == 0.0 and iv_change == 0.0:
        entry_trigger = "⏳ WAITING — enter live spot + open price to unlock triggers"
        trigger_cls   = "regime-purple"
    else:
        entry_trigger = "⚠ MIXED SIGNALS — wait for price + IV confirmation"
        trigger_cls   = "regime-purple"

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
        # Execution trigger
        "Entry Trigger":        entry_trigger,
        "Trigger Class":        trigger_cls,
        # IV state
        "IV Change":            round(iv_change * 100, 2),   # in pct points
        "Price Change":         round(price_change, 2),
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
@st.cache_data(ttl=1800, show_spinner=False)
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
        t, hist = _yf_history_with_retry(gold_spot_sym, period="5d")
        if t is not None and not hist.empty:
            p = float(hist["Close"].dropna().iloc[-1])
            if p > 1500:
                gc_spot = p
                break
    if gc_spot is None:
        # Last resort: GLD ETF × ~10.5 ratio
        try:
            _, hist = _yf_history_with_retry("GLD", period="2d")
            if not hist.empty:
                gc_spot = round(float(hist["Close"].dropna().iloc[-1]) * 10.5, 1)
        except Exception:
            pass
    if gc_spot is None:
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
        df_gld, gc_spot, gc_expiry, etf_spot, scale, proxy_sym = _gld_proxy(gc_spot, authoritative_spot=gc_spot)
        gold_raw    = df_gld
        gold_source = f"{proxy_sym} Proxy  (${etf_spot:.2f} × {scale:.2f})"
    else:
        # ── STRIKE SANITY GUARD ───────────────────────────────────────────────
        # Problem: CME returns deferred-contract strikes (~$4,100–$4,700) when
        # the front-month has expired. gc_spot can also be wrong (yfinance returns
        # GC futures close ~$3,960 instead of XAU=X ~$3,320), so we cannot
        # anchor the filter to gc_spot.
        #
        # Solution: use a HARD RANGE for gold that is instrument-specific and
        # reality-anchored. Gold has not been above $4,000 as of June 2026.
        # Any chain with median strike > $4,000 is a deferred contract.
        # Secondary check: if the user entered an XAU override, use that as anchor.
        GOLD_HARD_MAX = 4000.0   # update this if gold genuinely breaks $4k
        GOLD_HARD_MIN = 1800.0

        strike_median = float(gold_raw["strike"].median())
        anchor = xau_override if xau_override > 1500 else gc_spot

        # Reject if median is above hard ceiling OR > 20% away from any valid anchor
        anchor_ok  = anchor > 1500
        too_high   = strike_median > GOLD_HARD_MAX
        far_from_anchor = anchor_ok and (abs(strike_median - anchor) / anchor > 0.20)

        if too_high or far_from_anchor:
            st.warning(
                f"⚠️ CME strike median (${strike_median:,.0f}) looks like a deferred "
                f"contract (spot ~${anchor:,.0f}). Switching to GLD proxy."
            )
            df_gld, gc_spot, gc_expiry, etf_spot, scale, proxy_sym = _gld_proxy(
                gc_spot, authoritative_spot=gc_spot)
            gold_raw    = df_gld
            gold_source = f"{proxy_sym} Proxy  (${etf_spot:.2f} × {scale:.2f})  [CME mismatch]"
        else:
            # Filter to strikes within ±25% of anchor (clean up sparse far wings)
            lo = max(GOLD_HARD_MIN, anchor * 0.75) if anchor_ok else GOLD_HARD_MIN
            hi = min(GOLD_HARD_MAX, anchor * 1.25) if anchor_ok else GOLD_HARD_MAX
            centred = gold_raw[(gold_raw["strike"] >= lo) & (gold_raw["strike"] <= hi)]
            if len(centred) >= 10:
                gold_raw = centred.reset_index(drop=True)

    # If user provided a CFD override, re-scale strikes to that price space.
    # Greeks stay in native option space; only the display conversion changes.
    if xau_override > 1500:
        xau_display = xau_override
        cfd_scale = xau_display / gc_spot
        gold_raw_display = gold_raw.copy()
        gold_raw_display["strike"] = (gold_raw_display["strike"] * cfd_scale).round(1)
        _gold_dte = dte(gc_expiry)
        gold_df = enrich(gold_raw_display, xau_display, GC_MULTIPLIER, _gold_dte)
        gold_atm_iv = get_atm_iv(gold_df, xau_display)
        gold_lvl, gold_filt = levels(gold_df, xau_display, "XAUUSD", days=_gold_dte)
        gold_block = backquant_block("XAUUSD", gold_lvl, xau_display, gc_expiry)
        gold_source += f"  |  ⚡ CFD override: {xau_display:.1f}"
    else:
        xau_display = gc_spot
        _gold_dte = dte(gc_expiry)
        gold_df = enrich(gold_raw, gc_spot, GC_MULTIPLIER, _gold_dte)
        gold_atm_iv = get_atm_iv(gold_df, gc_spot)
        gold_lvl, gold_filt = levels(gold_df, gc_spot, "XAUUSD", days=_gold_dte)
        gold_block = backquant_block("XAUUSD", gold_lvl, gc_spot, gc_expiry)

    results["gold"] = {
        "lvl": gold_lvl, "filt": gold_filt, "spot": xau_display,
        "auto_spot": gc_spot, "atm_iv": gold_atm_iv,
        "expiry": gc_expiry, "source": gold_source, "block": gold_block,
    }

    # ── SPY / US500 ───────────────────────────────────────────────────────────
    # Fetch SPX index spot first — this is the real US500 price (FXCM CFD tracks ^GSPC)
    spx = None
    try:
        _, h = _yf_history_with_retry("^GSPC", period="2d")
        if not h.empty:
            spx = float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass

    spy, spy_hist = _yf_history_with_retry("SPY", period="2d")
    if spy is None or spy_hist.empty:
        raise RuntimeError("SPY data unavailable after retries — US500 levels cannot be computed.")
    s_spot = float(spy_hist["Close"].dropna().iloc[-1])

    s_exp = spy.options[0]
    sch   = _yf_option_chain_with_retry(spy, s_exp)
    if sch is None:
        raise RuntimeError("SPY option chain unavailable after retries.")
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
    s_atm_iv = get_atm_iv(s_rich, spx_display)
    s_lvl, s_filt = levels(s_rich, spx_display, "US500", days=_s_dte)
    s_block  = backquant_block("US500", s_lvl, spx_display, s_exp)

    results["spy"] = {
        "lvl": s_lvl, "filt": s_filt, "spot": s_spot, "index_spot": spx_display,
        "auto_index": spx, "atm_iv": s_atm_iv,
        "expiry": s_exp, "scale": spx_scale, "block": s_block,
    }

    # ── QQQ / US100 ───────────────────────────────────────────────────────────
    # Fetch NDX index spot first — this is the real US100 price (FXCM CFD tracks ^NDX)
    ndx = None
    try:
        _, h = _yf_history_with_retry("^NDX", period="2d")
        if not h.empty:
            ndx = float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass

    qqq, qqq_hist = _yf_history_with_retry("QQQ", period="2d")
    if qqq is None or qqq_hist.empty:
        raise RuntimeError("QQQ data unavailable after retries — US100 levels cannot be computed.")
    q_spot = float(qqq_hist["Close"].dropna().iloc[-1])

    q_exp = qqq.options[0]
    qch   = _yf_option_chain_with_retry(qqq, q_exp)
    if qch is None:
        raise RuntimeError("QQQ option chain unavailable after retries.")
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
    q_atm_iv = get_atm_iv(q_rich, ndx_display)
    q_lvl, q_filt = levels(q_rich, ndx_display, "US100", days=_q_dte)
    q_block  = backquant_block("US100", q_lvl, ndx_display, q_exp)

    results["qqq"] = {
        "lvl": q_lvl, "filt": q_filt, "spot": q_spot, "index_spot": ndx_display,
        "auto_index": ndx, "atm_iv": q_atm_iv,
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

      <div class="section-divider"></div>
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
      <div class="level-row" style="padding:3px 0;">
        <span class="level-label">Charm Signal</span>
        <span class="regime-orange" style="font-size:0.8rem;">{lvl['Charm Note']}</span>
      </div>
      <div class="level-row" style="padding:6px 0 8px 0;border-top:1px solid #1a3050;margin-top:4px;">
        <span class="level-label">Entry Trigger</span>
        <span class="{lvl['Trigger Class']}" style="font-size:0.82rem;text-align:right;max-width:72%;">{lvl['Entry Trigger']}</span>
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

st.markdown("""
<div style="background:#0d1117;border:1px solid #2a1a3a;border-radius:8px;padding:10px 16px 6px 16px;
            font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a6070;margin-bottom:12px;margin-top:8px;">
  📍 SESSION OPEN PRICES &nbsp;—&nbsp; enter today's open to enable Vanna + entry trigger logic
</div>
""", unsafe_allow_html=True)

op_col1, op_col2, op_col3 = st.columns(3)
with op_col1:
    xau_open = st.number_input(
        "XAUUSD open price", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Today's session open for XAUUSD. Used to compute price_change for Vanna logic.")
with op_col2:
    us500_open = st.number_input(
        "US500 open price", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Today's session open for US500. Used to compute price_change for Vanna logic.")
with op_col3:
    us100_open = st.number_input(
        "US100 open price", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Today's session open for US100. Used to compute price_change for Vanna logic.")

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

    # ── Live regime re-computation (outside cache) ────────────────────────────
    # fetch_all is cached (data is static). Regime engine needs live inputs:
    # ATM IV vs session-open IV, and price vs open price.
    # We re-run levels() with live iv_change + price_change — fast, no network.

    def recompute_levels(instrument_key, filt_df, spot, open_price, atm_iv, days_val, label):
        open_iv, iv_change, iv_change_pct = store_session_open_iv(instrument_key, atm_iv)
        price_change = (spot - open_price) if open_price > 0 else 0.0
        lvl, _ = levels(filt_df, spot, label, days=days_val,
                        iv_change=iv_change, price_change=price_change)
        return lvl, open_iv, iv_change_pct

    g = data["gold"]
    _g_dte = dte(g["expiry"])
    g_lvl, g_open_iv, g_iv_pct = recompute_levels(
        "xau", g["filt"], g["spot"], xau_open, g["atm_iv"], _g_dte, "XAUUSD")

    s = data["spy"]
    _s_dte = dte(s["expiry"])
    s_lvl, s_open_iv, s_iv_pct = recompute_levels(
        "us500", s["filt"], s["index_spot"], us500_open, s["atm_iv"], _s_dte, "US500")

    q = data["qqq"]
    _q_dte = dte(q["expiry"])
    q_lvl, q_open_iv, q_iv_pct = recompute_levels(
        "us100", q["filt"], q["index_spot"], us100_open, q["atm_iv"], _q_dte, "US100")

    st.success(f"Levels ready — {datetime.now().strftime('%d %b %Y  %H:%M')}")

    tab1, tab2, tab3 = st.tabs(["🥇 XAUUSD", "📈 US500", "💻 US100"])

    with tab1:
        badge_cls = "badge-cme" if "CME" in g["source"] else "badge-gld"
        spot_note = f"  |  auto: {g['auto_spot']:.1f}" if xau_override > 1500 else f"  auto: {g['auto_spot']:.1f}"
        st.markdown(f'<span class="source-badge {badge_cls}">SOURCE: {g["source"]}{spot_note}</span>',
                    unsafe_allow_html=True)
        iv_dir = "↓" if g_iv_pct < 0 else "↑"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#4a6070;margin-bottom:8px;">'
            f'ATM IV: <b style="color:#ce93d8">{g["atm_iv"]*100:.1f}%</b> &nbsp;|&nbsp; '
            f'Open IV: <b style="color:#8a9bb0">{g_open_iv*100:.1f}%</b> &nbsp;|&nbsp; '
            f'IV Change: <b style="color:{"#ff5252" if g_iv_pct > 0 else "#00e676"}">{iv_dir}{abs(g_iv_pct):.1f}%</b>'
            f'</div>', unsafe_allow_html=True)
        render_level_card(g_lvl)
        st.markdown("**📋 BackQuant Paste Block — copy and paste into indicator settings:**")
        st.code(backquant_block("XAUUSD", g_lvl, g["spot"], g["expiry"]), language=None)
        st.pyplot(gex_chart(g["filt"], g_lvl, g["spot"], "XAUUSD"))

    with tab2:
        override_tag = "  ⚡ override" if us500_override > 1000 else ""
        auto_note    = f"  auto: {s['auto_index']:.1f}" if us500_override > 1000 else ""
        st.markdown(f'<span class="source-badge badge-cme">US500 {s["index_spot"]:.1f}{override_tag}  (SPY×{s["scale"]:.4f}){auto_note}</span>',
                    unsafe_allow_html=True)
        iv_dir = "↓" if s_iv_pct < 0 else "↑"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#4a6070;margin-bottom:8px;">'
            f'ATM IV: <b style="color:#ce93d8">{s["atm_iv"]*100:.1f}%</b> &nbsp;|&nbsp; '
            f'Open IV: <b style="color:#8a9bb0">{s_open_iv*100:.1f}%</b> &nbsp;|&nbsp; '
            f'IV Change: <b style="color:{"#ff5252" if s_iv_pct > 0 else "#00e676"}">{iv_dir}{abs(s_iv_pct):.1f}%</b>'
            f'</div>', unsafe_allow_html=True)
        render_level_card(s_lvl)
        st.markdown("**📋 BackQuant Paste Block:**")
        st.code(backquant_block("US500", s_lvl, s["index_spot"], s["expiry"]), language=None)
        st.pyplot(gex_chart(s["filt"], s_lvl, s["index_spot"], "US500"))

    with tab3:
        override_tag = "  ⚡ override" if us100_override > 5000 else ""
        auto_note    = f"  auto: {q['auto_index']:.1f}" if us100_override > 5000 else ""
        st.markdown(f'<span class="source-badge badge-cme">US100 {q["index_spot"]:.1f}{override_tag}  (QQQ×{q["scale"]:.4f}){auto_note}</span>',
                    unsafe_allow_html=True)
        iv_dir = "↓" if q_iv_pct < 0 else "↑"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#4a6070;margin-bottom:8px;">'
            f'ATM IV: <b style="color:#ce93d8">{q["atm_iv"]*100:.1f}%</b> &nbsp;|&nbsp; '
            f'Open IV: <b style="color:#8a9bb0">{q_open_iv*100:.1f}%</b> &nbsp;|&nbsp; '
            f'IV Change: <b style="color:{"#ff5252" if q_iv_pct > 0 else "#00e676"}">{iv_dir}{abs(q_iv_pct):.1f}%</b>'
            f'</div>', unsafe_allow_html=True)
        render_level_card(q_lvl)
        st.markdown("**📋 BackQuant Paste Block:**")
        st.code(backquant_block("US100", q_lvl, q["index_spot"], q["expiry"]), language=None)
        st.pyplot(gex_chart(q["filt"], q_lvl, q["index_spot"], "US100"))

    st.markdown("""
    <div class="footer">
      Levels cached 1h · Regime engine live (ATM IV · session-open anchor) · CME → GLD fallback<br>
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
