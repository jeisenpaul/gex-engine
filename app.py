"""
GEX Engine â€” US500 / US100
Streamlit app. Gold removed â€” yfinance SPY/QQQ only.
Run: streamlit run app.py
Deploy free: streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf
import warnings
from scipy.stats import norm
from datetime import datetime, date, timedelta
from html import escape

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

warnings.filterwarnings("ignore")

# â”€â”€ PAGE CONFIG â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="GEX Engine",
    page_icon="ðŸ“Š",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# â”€â”€ STYLING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
.badge-src  { background: #0d2a0d; color: #00e676; border: 1px solid #00e676; }

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

# â”€â”€ CONSTANTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RISK_FREE_RATE        = 0.05
EQ_MULTIPLIER         = 100

REQUIRED_CHAIN_COLUMNS = ["strike", "call_oi", "call_iv", "put_oi", "put_iv"]


def market_now(tz_name="America/New_York"):
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(tz_name))
    try:
        import pytz
        return datetime.now(pytz.timezone(tz_name))
    except Exception:
        return datetime.now()


def first_valid_close(ticker_symbol, period="5d"):
    history = yf.Ticker(ticker_symbol).history(period=period, auto_adjust=False)
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError(f"No recent close returned for {ticker_symbol}.")
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if closes.empty:
        raise RuntimeError(f"No valid close returned for {ticker_symbol}.")
    return float(closes.iloc[-1])


def first_expiry(ticker_obj, ticker_symbol):
    expiries = list(getattr(ticker_obj, "options", []) or [])
    if not expiries:
        raise RuntimeError(f"No option expiries returned for {ticker_symbol}.")
    return expiries[0]


def option_chain_frame(ticker_obj, expiry, ticker_symbol):
    chain = ticker_obj.option_chain(expiry)
    calls = getattr(chain, "calls", pd.DataFrame())
    puts = getattr(chain, "puts", pd.DataFrame())
    needed = ["strike", "openInterest", "impliedVolatility"]
    if calls.empty or puts.empty:
        raise RuntimeError(f"Empty option chain returned for {ticker_symbol} {expiry}.")
    missing_calls = [col for col in needed if col not in calls.columns]
    missing_puts = [col for col in needed if col not in puts.columns]
    if missing_calls or missing_puts:
        raise RuntimeError(f"Incomplete option chain returned for {ticker_symbol} {expiry}.")

    df = pd.merge(
        calls[needed].rename(
            columns={"openInterest": "call_oi", "impliedVolatility": "call_iv"}),
        puts[needed].rename(
            columns={"openInterest": "put_oi", "impliedVolatility": "put_iv"}),
        on="strike",
        how="outer",
    )
    for col in REQUIRED_CHAIN_COLUMNS:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
    df = df[df["strike"] > 0].sort_values("strike").reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No usable strikes returned for {ticker_symbol} {expiry}.")
    df["call_settle"] = 0.0
    df["put_settle"] = 0.0
    return df


def build_scaled_index(symbol, index_symbol, fallback_scale, override, override_min,
                       display_label):
    try:
        index_spot = first_valid_close(index_symbol)
    except Exception:
        index_spot = None

    ticker = yf.Ticker(symbol)
    etf_spot = first_valid_close(symbol)
    expiry = first_expiry(ticker, symbol)
    raw_df = option_chain_frame(ticker, expiry, symbol)

    if index_spot is None:
        index_spot = etf_spot * fallback_scale

    display_spot = float(override) if override > override_min else float(index_spot)
    if display_spot <= 0 or etf_spot <= 0:
        raise RuntimeError(f"Invalid spot data for {display_label}.")

    scale = display_spot / etf_spot
    scaled_df = raw_df.copy()
    scaled_df["strike"] = (scaled_df["strike"] * scale).round(1)
    days = dte(expiry)
    rich = enrich(scaled_df, display_spot, EQ_MULTIPLIER, days)
    atm_iv = get_atm_iv(rich, display_spot)
    lvl, filt = levels(rich, display_spot, display_label, days=days)
    block = backquant_block(display_label, lvl, display_spot, expiry)

    return {
        "lvl": lvl,
        "filt": filt,
        "spot": etf_spot,
        "index_spot": display_spot,
        "auto_index": float(index_spot),
        "atm_iv": atm_iv,
        "expiry": expiry,
        "scale": round(scale, 4),
        "block": block,
    }

# â”€â”€ MATHS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    Dealers short calls â†’ long vanna â†’ buy spot when vol drops, sell when vol rises.
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
    Extract ATM implied volatility â€” the strike closest to spot.
    This is the correct IV input for Vanna and regime logic.
    Chain-average IV is distorted by deep OTM strikes; ATM IV reflects
    what dealers are actually pricing for near-money hedging.
    Returns the average of ATM call_iv and put_iv (put-call ATM average).
    """
    if df is None or df.empty or spot <= 0:
        return 0.20
    clean = df.copy()
    clean["strike"] = pd.to_numeric(clean.get("strike", 0), errors="coerce")
    clean = clean.dropna(subset=["strike"])
    if clean.empty:
        return 0.20
    atm_idx = (clean["strike"] - spot).abs().idxmin()
    civ = float(clean.loc[atm_idx, "call_iv"]) if "call_iv" in clean.columns else 0.0
    piv = float(clean.loc[atm_idx, "put_iv"])  if "put_iv"  in clean.columns else 0.0
    # Fall back to enriched columns if raw IV is zero
    if civ <= 0.01 and "call_iv_c" in df.columns:
        civ = float(clean.loc[atm_idx, "call_iv_c"])
    if piv <= 0.01 and "put_iv_c" in df.columns:
        piv = float(clean.loc[atm_idx, "put_iv_c"])
    atm = (civ + piv) / 2 if (civ > 0.01 and piv > 0.01) else max(civ, piv)
    return atm if atm > 0.01 else 0.20

def store_session_open_iv(key, current_iv):
    """
    Store IV at session open â€” NOT first refresh IV.
    Uses NY market open time (09:30 ET) as the reference point.
    If it is before market open or a new trading day, resets the stored IV.
    Returns (open_iv, iv_change, iv_change_pct).
    """
    now_ny   = market_now("America/New_York")
    today_str = now_ny.strftime("%Y-%m-%d")
    open_key  = f"{key}_open_iv"
    date_key  = f"{key}_open_date"
    hour_ny   = now_ny.hour + now_ny.minute / 60

    stored_date = st.session_state.get(date_key, "")
    stored_iv   = st.session_state.get(open_key, None)

    # Reset if: new trading day OR before 09:30 (pre-market â€” don't anchor to pre-market IV)
    is_new_day    = stored_date != today_str
    is_premarket  = hour_ny < 9.5
    is_afterhours = hour_ny >= 17.0

    if is_new_day or stored_iv is None:
        if not is_premarket and not is_afterhours:
            # Market is open â€” anchor to this IV as today's open reference
            st.session_state[open_key]  = current_iv
            st.session_state[date_key]  = today_str
            stored_iv = current_iv
        else:
            # Pre-market â€” don't store yet, no reliable anchor
            stored_iv = current_iv

    open_iv        = stored_iv
    iv_change      = current_iv - open_iv
    iv_change_pct  = (iv_change / open_iv * 100) if open_iv > 0 else 0.0
    return open_iv, iv_change, iv_change_pct


def dte(expiry_str):
    try:
        e = datetime.strptime(expiry_str[:10], "%Y-%m-%d").date()
        return max((e - date.today()).days, 1)
    except: return 20

# â”€â”€ CORE COMPUTATION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        civ = row["call_iv"] if row.get("call_iv", 0) > 0.01 else 0.15
        piv = row["put_iv"]  if row.get("put_iv",  0) > 0.01 else (civ if civ > 0.01 else 0.15)
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

    # GEX: gamma Ã— OI Ã— multiplier Ã— SÂ²
    df["call_gex"]  =  df["call_gamma"] * df["call_oi"] * multiplier * spot**2
    df["put_gex"]   = -df["put_gamma"]  * df["put_oi"]  * multiplier * spot**2
    df["net_gex"]   =  df["call_gex"]   + df["put_gex"]

    # DEX: delta Ã— OI Ã— multiplier Ã— S
    df["call_dex"]  =  df["call_delta"] * df["call_oi"] * multiplier * spot
    df["put_dex"]   =  df["put_delta"]  * df["put_oi"]  * multiplier * spot
    df["net_dex"]   =  df["call_dex"]   + df["put_dex"]

    # VEX (Vanna Exposure): vanna Ã— OI Ã— multiplier Ã— S
    # Dealers short calls â†’ positive call_vanna position â†’ buy spot when IV drops
    # Net VEX > 0 = vol-down is bullish (dealers buy spot); < 0 = vol-down is bearish
    df["call_vex"]  =  df["call_vanna"] * df["call_oi"] * multiplier * spot
    df["put_vex"]   = -df["put_vanna"]  * df["put_oi"]  * multiplier * spot
    df["net_vex"]   =  df["call_vex"]   + df["put_vex"]

    # CEX (Charm Exposure): per-day delta units from dealer perspective
    # Dealers SHORT calls  â†’ charm exposure = -(call_charm Ã— OI) â†’ they LOSE delta each day on OTM calls
    # Dealers LONG puts    â†’ charm exposure = -(put_charm Ã— OI)  â†’ put charm is negative for OTM puts
    # Net positive CEX = dealers need to BUY spot each day to stay hedged (bullish daily pressure)
    # Net negative CEX = dealers need to SELL spot each day (bearish daily pressure)
    df["call_cex"]  = -df["call_charm"] * df["call_oi"] * multiplier * spot / 365
    df["put_cex"]   =  df["put_charm"]  * df["put_oi"]  * multiplier * spot / 365
    df["net_cex"]   =  df["call_cex"]   + df["put_cex"]

    return df

def levels(df, spot, label="", days=30, iv_change=0.0, price_change=0.0,
           filter_pct=0.15, has_open_price=False):
    if df is None or df.empty:
        raise ValueError(f"No option data available for {label or 'instrument'}.")
    df = df.copy()
    for col in REQUIRED_CHAIN_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df[df["strike"] > 0].sort_values("strike").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No valid strikes available for {label or 'instrument'}.")
    # Primary strike window is instrument-specific (gold: Â±10%, equities: Â±15%).
    # Far-OTM strikes inflate GEX Flip, DEX walls, and Vanna â€” cap them.
    # Fall back to progressively wider ranges only if we can't find â‰¥10 strikes.
    filt = None
    fallback_pcts = sorted(set([filter_pct, filter_pct+0.05, 0.30, 0.50, 1.0]))
    for pct in fallback_pcts:
        mask = (df["strike"] >= spot*(1-pct)) & (df["strike"] <= spot*(1+pct))
        candidate = df[mask].sort_values("strike").reset_index(drop=True)
        if len(candidate) >= 10:
            filt = candidate
            break
    if filt is None or filt.empty:
        filt = df.sort_values("strike").reset_index(drop=True)

    # â”€â”€ GEX levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Three DISTINCT concepts â€” these should almost never be the same strike:
    #
    # HVL (High Volatility Level) = strike with highest ABSOLUTE net_gex.
    #   This is the dealer pinning anchor / gamma gravity centre.
    #   It can be above or below spot.
    #
    # Call Wall = strike with highest positive call_gex ABOVE spot (resistance).
    #   Dealers are short calls here â€” heavy buying if price approaches.
    #
    # Put Wall  = strike with most negative put_gex BELOW spot (support).
    #   Dealers are long puts here â€” heavy selling if price approaches from above.
    #
    # Using separate call_gex / put_gex prevents the sign-inversion bug where
    # combined net_gex places both walls on the same side of spot.

    net_gex_by_strike  = filt.groupby("strike")["net_gex"].sum()
    call_gex_by_strike = filt.groupby("strike")["call_gex"].sum()
    put_gex_by_strike  = filt.groupby("strike")["put_gex"].sum()

    # HVL â€” highest absolute net_gex, any side
    hvl = float(net_gex_by_strike.abs().idxmax())

    # Call Wall â€” highest positive call_gex strictly above spot
    above_mask = call_gex_by_strike.index >= spot
    below_mask = put_gex_by_strike.index  <= spot

    if above_mask.any():
        call_wall = float(call_gex_by_strike[above_mask].idxmax())
    else:
        # Fallback: best call_gex anywhere (sparse chain)
        call_wall = float(call_gex_by_strike.idxmax())

    if below_mask.any():
        put_wall = float(put_gex_by_strike[below_mask].idxmin())
    else:
        # Fallback: most negative put_gex anywhere (sparse chain)
        put_wall = float(put_gex_by_strike.idxmin())

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
    # Clamp GEX Flip between Put Wall and Call Wall
    gflip = float(np.clip(gflip, min(put_wall, call_wall), max(put_wall, call_wall)))

    # â”€â”€ DEX levels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Call DEX Wall = strike where call delta exposure peaks (above spot)
    # Put DEX Wall  = strike where put delta exposure is most negative (below spot)
    call_dex_by_strike = filt.groupby("strike")["call_dex"].sum()
    put_dex_by_strike  = filt.groupby("strike")["put_dex"].sum()
    call_dex_above = call_dex_by_strike[call_dex_by_strike.index >= spot]
    put_dex_below = put_dex_by_strike[put_dex_by_strike.index <= spot]
    call_delta_wall = float(call_dex_above.idxmax()) if not call_dex_above.empty else float(call_dex_by_strike.idxmax())
    put_delta_wall  = float(put_dex_below.idxmin()) if not put_dex_below.empty else float(put_dex_by_strike.idxmin())

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

    # â”€â”€ VEX levels (Vanna Flip) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Max Pain â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Use filtered strikes â€” full df includes far-OTM strikes with noise OI
    # that pull Max Pain away from the tradeable zone
    sa  = filt["strike"].values
    coi = filt["call_oi"].values.astype(float)
    poi = filt["put_oi"].values.astype(float)
    lss = [np.sum(np.maximum(sa-s,0)*coi)+np.sum(np.maximum(s-sa,0)*poi) for s in sa]
    max_pain = float(sa[np.argmin(lss)])

    # â”€â”€ Scalar exposures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    total_net_gex = float(filt["net_gex"].sum())
    total_net_dex = float(filt["net_dex"].sum())
    total_net_vex = float(filt["net_vex"].sum())
    total_net_cex = float(filt["net_cex"].sum())   # per-day delta units

    # â”€â”€ Regime labels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    gex_positive = spot > gflip
    regime       = "POSITIVE â–²" if gex_positive else "NEGATIVE â–¼"

    dex_positive = total_net_dex > 0
    dex_bias     = "BULLISH â–²" if dex_positive else "BEARISH â–¼"

    # â”€â”€ Vanna regime â€” price Ã— IV direction (document Issue 4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # The document explicitly states:
    #   Price â†‘ + IV â†“ = true bullish Vanna (squeeze)
    #   Price â†“ + IV â†‘ = true bearish Vanna (crash acceleration)
    # We also need net VEX sign to know which direction IV pressure moves dealers.
    # Combined: price direction + iv direction + net VEX sign.
    vex_positive   = total_net_vex > 0
    price_up       = price_change >= 0
    iv_falling     = iv_change <= 0

    # True bullish Vanna squeeze: price up AND IV falling
    # True bearish Vanna:         price down AND IV rising
    # Neutral / mixed:            price and IV moving same direction
    if price_up and iv_falling:
        # Classic squeeze condition â€” confirm with VEX
        if not vex_positive:
            # VEX < 0: dealers net short vanna â†’ IV drop forces them to BUY spot â†’ amplifies squeeze
            vanna_regime = "VOL-CRUSH BULLISH â–²"
            vanna_note   = "âš¡ PRICEâ†‘ + IVâ†“ + VEXâˆ’ = SQUEEZE FUEL"
        else:
            # VEX > 0: dealers net long vanna â†’ IV drop forces them to SELL â†’ squeeze is fading
            vanna_regime = "VOL-CRUSH BEARISH â–¼"
            vanna_note   = "âš  PRICEâ†‘ + IVâ†“ + VEX+ = RALLY FADING"
    elif not price_up and not iv_falling:
        # Classic crash acceleration â€” confirm with VEX
        if vex_positive:
            # VEX > 0: dealers net long vanna â†’ IV spike forces them to BUY â†’ provides floor
            vanna_regime = "VOL-SPIKE BULLISH â–²"
            vanna_note   = "âš  PRICEâ†“ + IVâ†‘ + VEX+ = FLOOR POSSIBLE"
        else:
            # VEX < 0: dealers net short vanna â†’ IV spike forces them to SELL â†’ crash accelerator
            vanna_regime = "VOL-SPIKE BEARISH â–¼"
            vanna_note   = "âš¡ PRICEâ†“ + IVâ†‘ + VEXâˆ’ = CRASH ACCELERATOR"
    elif price_up and not iv_falling:
        # Price up but IV also rising â€” dealers under pressure, unstable; NOT a clean squeeze
        # Rising IV with rising price = real fear + FOMO combo; fade risk is high
        vanna_regime = "VOL-SPIKE UNSTABLE âš "
        vanna_note   = f"âš  PRICEâ†‘ + IVâ†‘ â€” UNSTABLE Â· WATCH REVERSAL"
    else:
        # Price down but IV also falling â€” dead range / exhaustion / coil
        # Dealers not hedging hard; expect range-bound or reversal trigger
        vanna_regime = "VOL-CRUSH NEUTRAL â±"
        vanna_note   = f"â± PRICEâ†“ + IVâ†“ â€” RANGE / EXHAUSTION Â· COIL FORMING"

    # â”€â”€ Charm regime â€” DTE-weighted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cex_positive   = total_net_cex > 0
    charm_flow     = "BULLISH BLEED â–²" if cex_positive else "BEARISH BLEED â–¼"
    charm_reliable = days < 21

    if charm_reliable:
        charm_note = "ðŸ“ˆ AFTERNOON MELT-UP BIAS" if cex_positive else "ðŸ“‰ AFTERNOON FADE BIAS"
    else:
        charm_note = f"â± CHARM: {'â†‘' if cex_positive else 'â†“'} (low weight â€” {days}+ DTE)"

    # â”€â”€ Combined Regime Engine â€” hierarchical â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # CORE PRINCIPLE (from audit):
    #   GEX sign  = VOLATILITY STRUCTURE (positive = pinning, negative = expansion)
    #   GEX sign â‰  direction
    #
    #   Direction is determined by: Vanna + IV + price + DEX
    #
    # HIERARCHY (evaluated top-down, first match wins):
    #
    # TIER 1 â€” FLOW OVERRIDE (Vanna + IV + price all aligned, any GEX sign)
    #   Strong Vanna + falling IV + rising price = squeeze bias regardless of gamma
    #   Strong Vanna + rising IV + falling price = crash/trend bias regardless of gamma
    #   When 3-of-3 flow signals agree, they override the GEX/DEX quadrant.
    #
    # TIER 2 â€” MIXED FLOW (Vanna aligned but price/IV partially conflicting)
    #   Use GEX to determine volatility character, DEX for directional lean.
    #
    # TIER 3 â€” STANDARD QUADRANTS (Vanna neutral/mixed, fall back to GEX+DEX)
    #   Classic four-quadrant model applies cleanly here.
    #
    # GEX sign modifies the STRATEGY (pinning vs expansion) but never blocks
    # a Vanna-confirmed directional classification.

    vanna_bullish = "BULLISH" in vanna_regime or "SQUEEZE FUEL" in vanna_note
    vanna_bearish = ("BEARISH" in vanna_regime or "CRASH ACCELERATOR" in vanna_note) \
                    and not vanna_bullish
    vanna_neutral = not vanna_bullish and not vanna_bearish

    # Flow signal agreement scores (3 = all aligned, 2 = majority, 0-1 = mixed)
    #
    # GUARD: Two distinct cases must be suppressed:
    #
    # Case A â€” user entered NO session open price at all (open_price == 0).
    #   price_change is forced to 0.0 by the caller (open_price > 0 check).
    #   We detect this via price_change == 0.0 AND iv_change == 0.0 together,
    #   but only when open_price was not provided (passed through as the sentinel 0).
    #   The iv_change == 0.0 alone is NOT a reliable guard â€” it can be legitimately
    #   0 on the first refresh of a real session (open_iv anchored to current_iv).
    #
    # Case B â€” price_change is exactly 0 AND iv_change is exactly 0.
    #   This is the degenerate case where both signals are flat. price_up = True
    #   (0 >= 0) and iv_falling = True (0 <= 0) would falsely inflate bull_signals.
    #   We suppress only when price_change is strictly zero AND we have no meaningful
    #   iv signal either (iv_change == 0.0), AND price_change is zero (no open price given).
    #
    # FIX: The guard must check that the USER actually entered a session open price.
    # We do this by passing open_price into the function and checking it directly.
    # Since open_price is not passed here, we infer: price_change == 0 means no open
    # was provided ONLY when iv_change is also 0 (i.e. first anchor = flat).
    # But if price_change != 0, the user DID enter an open price â€” never suppress.
    #
    # Use STRICT inequalities for price_up / iv_falling so that zero is neutral:
    price_up_strict   = price_change > 0     # strictly up â€” zero is NOT bullish
    iv_falling_strict = iv_change    < 0     # strictly falling â€” zero is NOT bullish
    price_dn_strict   = price_change < 0
    iv_rising_strict  = iv_change    > 0

    # Suppress Tier 1/2 only when user gave no real inputs at all
    no_live_input = (not has_open_price and iv_change == 0.0)

    bull_signals = 0 if no_live_input else sum([price_up_strict, iv_falling_strict, vanna_bullish])
    bear_signals = 0 if no_live_input else sum([price_dn_strict, iv_rising_strict,  vanna_bearish])

    # GEX describes volatility character for strategy text
    gex_char = "pinning" if gex_positive else "expansion"

    # â”€â”€ TIER 1: Strong flow override (all 3 aligned) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if bull_signals == 3:
        # Priceâ†‘ + IVâ†“ + Vanna bullish â€” squeeze regardless of GEX sign
        if gex_positive:
            # Positive GEX + squeeze flow = controlled melt-up, bounded by walls
            intraday_condition = "BULLISH SQUEEZE"
            strategy           = f"Buy pullbacks to GEX flip Â· Walls cap range Â· IV crush accelerating Â· GEX {gex_char}"
            condition_cls      = "regime-pos"
        else:
            # Negative GEX + squeeze flow = momentum expansion, no pinning resistance
            if dex_positive:
                intraday_condition = "SQUEEZE TREND â–²"
                strategy           = "Momentum longs Â· Breakout bias Â· No gamma ceiling Â· Trail stops up"
                condition_cls      = "regime-orange"
            else:
                intraday_condition = "SQUEEZE RISK â–²"
                strategy           = "Long bias Â· Breakout watch Â· DEX lagging flow Â· Confirm GEX flip reclaim"
                condition_cls      = "regime-orange"

    elif bear_signals == 3:
        # Priceâ†“ + IVâ†‘ + Vanna bearish â€” crash/trend regardless of GEX sign
        if gex_positive:
            # Positive GEX + bearish flow = range breakdown attempt
            intraday_condition = "BEARISH SQUEEZE"
            strategy           = f"Sell rallies to GEX flip Â· Watch for floor Â· IV spike Â· GEX {gex_char}"
            condition_cls      = "regime-neg"
        else:
            # Negative GEX + bearish flow = true crash/trend
            if not dex_positive:
                intraday_condition = "CRASH / TREND â–¼"
                strategy           = "Momentum shorts Â· Avoid catching bottoms Â· Trail stops down Â· No gamma floor"
                condition_cls      = "regime-neg"
            else:
                intraday_condition = "BREAKDOWN RISK â–¼"
                strategy           = "Short bias Â· DEX still holding Â· Watch for capitulation Â· Fade DEX bounces"
                condition_cls      = "regime-neg"

    # â”€â”€ TIER 2: Majority flow (2-of-3 aligned) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    elif bull_signals == 2:
        # Two bullish signals â€” directional lean is bullish, confirm with DEX
        if dex_positive:
            intraday_condition = "BULLISH LEAN"
            strategy           = f"Long bias Â· Flow supportive Â· GEX {gex_char} Â· Confirm price action"
            condition_cls      = "regime-pos"
        else:
            intraday_condition = "UNSTABLE BULLISH"
            strategy           = f"Cautious long Â· DEX lagging Â· GEX {gex_char} Â· Wait for DEX flip confirmation"
            condition_cls      = "regime-orange"

    elif bear_signals == 2:
        # Two bearish signals â€” directional lean is bearish, confirm with DEX
        if not dex_positive:
            intraday_condition = "BEARISH LEAN"
            strategy           = f"Short bias Â· Flow pressuring Â· GEX {gex_char} Â· Confirm breakdown"
            condition_cls      = "regime-neg"
        else:
            intraday_condition = "UNSTABLE BEARISH"
            strategy           = f"Cautious short Â· DEX holding Â· GEX {gex_char} Â· Wait for DEX capitulation"
            condition_cls      = "regime-orange"

    # â”€â”€ TIER 3: Mixed flow â€” fall back to GEX + DEX quadrants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    else:
        if gex_positive and dex_positive:
            intraday_condition = "STABLE BULLISH"
            strategy           = "Buy pullbacks Â· Fade resistance Â· Gamma pinning â€” avoid chasing"
            condition_cls      = "regime-pos"
        elif gex_positive and not dex_positive:
            intraday_condition = "STABLE BEARISH"
            strategy           = "Sell rallies Â· Fade bounces Â· Gamma pinning â€” avoid breakout longs"
            condition_cls      = "regime-neg"
        elif not gex_positive and dex_positive:
            intraday_condition = "SQUEEZE WATCH"
            strategy           = "Momentum long watch Â· Gamma expanding Â· Vanna mixed Â· Confirm IV direction"
            condition_cls      = "regime-orange"
        else:
            intraday_condition = "TREND / RANGE â–¼"
            strategy           = "Short bias Â· Gamma expanding Â· Vanna mixed Â· No clear squeeze setup"
            condition_cls      = "regime-neg"

    # â”€â”€ Execution triggers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Conditional entry filters â€” tell you WHEN the current regime is actionable.
    if bull_signals == 3 and dex_positive:
        entry_trigger = "âš¡ LONG: priceâ†‘ + IVâ†“ + Vanna bullish + DEX confirmed â€” squeeze active"
        trigger_cls   = "regime-pos" if gex_positive else "regime-orange"
    elif bull_signals == 3 and not dex_positive:
        entry_trigger = "âš¡ SQUEEZE LONG: priceâ†‘ + IVâ†“ + Vanna bullish Â· DEX lagging â€” wait for flip"
        trigger_cls   = "regime-orange"
    elif bear_signals == 3 and not dex_positive:
        entry_trigger = "âœ… SHORT: priceâ†“ + IVâ†‘ + Vanna bearish + DEX confirmed â€” trend active"
        trigger_cls   = "regime-neg"
    elif bear_signals == 3 and dex_positive:
        entry_trigger = "âš  BREAKDOWN RISK: priceâ†“ + IVâ†‘ + Vanna bearish Â· DEX still holding"
        trigger_cls   = "regime-neg"
    elif bull_signals == 2 and dex_positive and iv_falling_strict:
        entry_trigger = "âœ… LONG LEAN: majority bullish flow + DEX confirmed Â· watch IV continuation"
        trigger_cls   = "regime-pos"
    elif bear_signals == 2 and not dex_positive and iv_rising_strict:
        entry_trigger = "âœ… SHORT LEAN: majority bearish flow + DEX confirmed Â· watch IV continuation"
        trigger_cls   = "regime-neg"
    elif price_change == 0.0 and iv_change == 0.0:
        entry_trigger = "â³ WAITING â€” enter live spot + open price to unlock triggers"
        trigger_cls   = "regime-purple"
    else:
        entry_trigger = "âš  MIXED SIGNALS â€” 2-of-3 flow signals needed for entry"
        trigger_cls   = "regime-purple"

    return {
        # GEX
        "HVL":                  round(hvl,                1),   # gamma gravity centre
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
        # â”€â”€ DEBUG STATE (always populated for diagnostics) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "_debug": {
            "price_change":     round(price_change, 4),
            "iv_change":        round(iv_change * 100, 4),   # pct points
            "price_up_strict":  price_up_strict,
            "iv_falling_strict":iv_falling_strict,
            "vanna_bullish":    vanna_bullish,
            "vanna_bearish":    vanna_bearish,
            "vanna_neutral":    vanna_neutral,
            "bull_signals":     bull_signals,
            "bear_signals":     bear_signals,
            "no_live_input":    no_live_input,
            "gex_positive":     gex_positive,
            "dex_positive":     dex_positive,
            "vex_positive":     vex_positive,
            "cex_positive":     cex_positive,
        },
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
        f"HVL (Gamma Anchor): ${lvl['HVL']:,.1f}\n"
        f"GEX Flip:           ${lvl['GEX Flip']:,.1f}\n"
        f"Call Resistance:    ${lvl['Call Wall']:,.1f}\n"
        f"Put Support:        ${lvl['Put Wall']:,.1f}\n"
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

# â”€â”€ CHART â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        (lvl["HVL"],        "#ffd740", "-",  2.0, f"HVL        {lvl['HVL']}"),
        (lvl["Call Wall"],  "#00e676", "-",  2.0, f"CALL WALL {lvl['Call Wall']}"),
        (lvl["Put Wall"],   "#ff5252", "-",  2.0, f"PUT WALL  {lvl['Put Wall']}"),
        (lvl["GEX Flip"],   "#00bcd4", "--", 1.5, f"GEX FLIP  {lvl['GEX Flip']}"),
        (lvl["Vanna Flip"], "#ce93d8", "-.", 1.5, f"VANNA FLIP {lvl['Vanna Flip']}"),
        (lvl["Max Pain"],   "#ffab40", ":",  1.2, f"MAX PAIN  {lvl['Max Pain']}"),
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

# â”€â”€ FETCH ALL DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all(_cache_buster: int = 0,
              us500_override: float = 0.0,
              us100_override: float = 0.0):
    results = {}
    results["spy"] = build_scaled_index(
        "SPY", "^GSPC", fallback_scale=10.0, override=us500_override,
        override_min=1000.0, display_label="US500",
    )
    results["qqq"] = build_scaled_index(
        "QQQ", "^NDX", fallback_scale=40.0, override=us100_override,
        override_min=5000.0, display_label="US100",
    )
    return results
# -- LEVEL CARD RENDERER -------------------------------------------------------
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
      <div class="section-header">â—ˆ INTRADAY REGIME ENGINE</div>
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
      <div class="level-row"><span class="level-label">HVL (Gamma Anchor)</span>
        <span class="level-val lvl-yellow">{lvl['HVL']:,.1f}</span></div>
      <div class="level-row"><span class="level-label">GEX Flip</span>
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
        <span class="level-val lvl-orange">{cex_str} Î”</span></div>

      <div class="section-divider"></div>
      <div class="section-header">CONTEXT</div>
      <div class="level-row"><span class="level-label">Max Pain</span>
        <span class="level-val lvl-yellow">{lvl['Max Pain']:,.1f}</span></div>
    </div>
    """, unsafe_allow_html=True)

# â”€â”€ DEBUG PANEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_debug_panel(lvl):
    """Collapsible signal state panel â€” shows exactly what the hierarchy sees."""
    d = lvl.get("_debug", {})
    if not d:
        return

    def tick(v):
        return "âœ…" if v else "âŒ"

    no_input_warn = "âš ï¸ no_live_input=True â†’ Tier 1/2 SUPPRESSED" if d.get("no_live_input") else "âœ… live inputs active"

    with st.expander("ðŸ” Signal Debug â€” hierarchy state", expanded=False):
        st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;line-height:2;color:#c9d1d9;
            background:#0d1117;border:1px solid #1e3a1e;border-radius:8px;padding:14px 18px;">
<b style="color:#ffd740">{no_input_warn}</b><br><br>

<b style="color:#4a6070">â”€â”€ RAW INPUTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€</b><br>
price_change    = <b style="color:#{'00e676' if d.get('price_change',0) > 0 else ('ff5252' if d.get('price_change',0) < 0 else '8a9bb0')}">{d.get('price_change', 'n/a')}</b><br>
iv_change       = <b style="color:#{'ff5252' if d.get('iv_change',0) > 0 else ('00e676' if d.get('iv_change',0) < 0 else '8a9bb0')}">{d.get('iv_change', 'n/a')}%</b><br><br>

<b style="color:#4a6070">â”€â”€ BOOLEAN SIGNALS (strict inequalities) â”€â”€</b><br>
price_up_strict    {tick(d.get('price_up_strict'))}   (price_change > 0)<br>
iv_falling_strict  {tick(d.get('iv_falling_strict'))}   (iv_change < 0)<br>
vanna_bullish      {tick(d.get('vanna_bullish'))}   (from Vanna regime label)<br>
vanna_bearish      {tick(d.get('vanna_bearish'))}<br>
vanna_neutral      {tick(d.get('vanna_neutral'))}<br><br>

<b style="color:#4a6070">â”€â”€ SIGNAL COUNTERS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€</b><br>
bull_signals = <b style="color:#{'00e676' if d.get('bull_signals',0)==3 else ('ffd740' if d.get('bull_signals',0)==2 else '#c9d1d9')}">{d.get('bull_signals', 0)}</b> / 3  â†’  Tier {'1 ðŸ”¥' if d.get('bull_signals')==3 else ('2' if d.get('bull_signals')==2 else '3 (fallback)')}<br>
bear_signals = <b style="color:#{'ff5252' if d.get('bear_signals',0)==3 else ('ffd740' if d.get('bear_signals',0)==2 else '#c9d1d9')}">{d.get('bear_signals', 0)}</b> / 3  â†’  Tier {'1 ðŸ”¥' if d.get('bear_signals')==3 else ('2' if d.get('bear_signals')==2 else '3 (fallback)')}<br><br>

<b style="color:#4a6070">â”€â”€ GEX/DEX/VEX FLAGS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€</b><br>
gex_positive  {tick(d.get('gex_positive'))}   (spot > GEX Flip)<br>
dex_positive  {tick(d.get('dex_positive'))}   (net DEX > 0)<br>
vex_positive  {tick(d.get('vex_positive'))}   (net VEX > 0)<br>
cex_positive  {tick(d.get('cex_positive'))}   (net CEX > 0)<br>
</div>
""", unsafe_allow_html=True)

# â”€â”€ UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<div class="gex-header">
  <h1>GEX ENGINE</h1>
  <p>US500 Â· US100 &nbsp;|&nbsp; GEX Â· DEX Â· Vanna Â· Charm &nbsp;|&nbsp; BackQuant Format</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#0d1117;border:1px solid #1a2a3a;border-radius:8px;padding:10px 16px 6px 16px;
            font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a6070;margin-bottom:12px;">
  âš¡ CFD SPOT OVERRIDES &nbsp;â€”&nbsp; leave at 0 to use auto-fetched prices (^GSPC Â· ^NDX)
</div>
""", unsafe_allow_html=True)

ov_col1, ov_col2 = st.columns(2)
with ov_col1:
    us500_override = st.number_input(
        "US500 CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Override auto-fetched ^GSPC. Set to 0 to use auto price.")
with ov_col2:
    us100_override = st.number_input(
        "US100 CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Override auto-fetched ^NDX. Set to 0 to use auto price.")

st.markdown("""
<div style="background:#0d1117;border:1px solid #2a1a3a;border-radius:8px;padding:10px 16px 6px 16px;
            font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a6070;margin-bottom:12px;margin-top:8px;">
  ðŸ“ SESSION OPEN PRICES &nbsp;â€”&nbsp; enter today's open to enable Vanna + entry trigger logic
</div>
""", unsafe_allow_html=True)

op_col1, op_col2 = st.columns(2)
with op_col1:
    us500_open = st.number_input(
        "US500 open price", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Today's session open for US500. Used to compute price_change for Vanna logic.")
with op_col2:
    us100_open = st.number_input(
        "US100 open price", min_value=0.0, value=0.0, step=0.1, format="%.1f",
        help="Today's session open for US100. Used to compute price_change for Vanna logic.")

btn_col, cc_col = st.columns([4, 1])
with btn_col:
    run = st.button("âš¡ Generate Levels")
with cc_col:
    if st.button("ðŸ—‘ Cache"):
        fetch_all.clear()
        st.success("Cache cleared")
        st.rerun()

# â”€â”€ RESULTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if run:
    with st.spinner("Fetching live spots and computing GEX + Vanna + Charm levelsâ€¦"):
        try:
            data = fetch_all(
                us500_override=us500_override,
                us100_override=us100_override,
            )
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    # â”€â”€ Live regime re-computation (outside cache) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # fetch_all is cached (data is static). Regime engine needs live inputs:
    # ATM IV vs session-open IV, and price vs open price.
    # We re-run levels() with live iv_change + price_change â€” fast, no network.

    def recompute_levels(instrument_key, filt_df, spot, open_price, atm_iv, days_val, label):
        open_iv, iv_change, iv_change_pct = store_session_open_iv(instrument_key, atm_iv)
        has_open_price = open_price > 0
        price_change = (spot - open_price) if has_open_price else 0.0
        lvl, _ = levels(filt_df, spot, label, days=days_val,
                        iv_change=iv_change, price_change=price_change,
                        has_open_price=has_open_price)
        return lvl, open_iv, iv_change_pct

    s = data["spy"]
    _s_dte = dte(s["expiry"])
    s_lvl, s_open_iv, s_iv_pct = recompute_levels(
        "us500", s["filt"], s["index_spot"], us500_open, s["atm_iv"], _s_dte, "US500")

    q = data["qqq"]
    _q_dte = dte(q["expiry"])
    q_lvl, q_open_iv, q_iv_pct = recompute_levels(
        "us100", q["filt"], q["index_spot"], us100_open, q["atm_iv"], _q_dte, "US100")

    st.success(f"Levels ready â€” {datetime.now().strftime('%d %b %Y  %H:%M')}")

    tab1, tab2 = st.tabs(["ðŸ“ˆ US500", "ðŸ’» US100"])

    with tab1:
        override_tag = "  âš¡ override" if us500_override > 1000 else ""
        auto_note    = f"  auto: {s['auto_index']:.1f}" if us500_override > 1000 else ""
        st.markdown(f'<span class="source-badge badge-src">US500 {s["index_spot"]:.1f}{override_tag}  (SPYÃ—{s["scale"]:.4f}){auto_note}</span>',
                    unsafe_allow_html=True)
        iv_dir = "â†“" if s_iv_pct < 0 else "â†‘"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#4a6070;margin-bottom:8px;">'
            f'ATM IV: <b style="color:#ce93d8">{s["atm_iv"]*100:.1f}%</b> &nbsp;|&nbsp; '
            f'Open IV: <b style="color:#8a9bb0">{s_open_iv*100:.1f}%</b> &nbsp;|&nbsp; '
            f'IV Change: <b style="color:{"#ff5252" if s_iv_pct > 0 else "#00e676"}">{iv_dir}{abs(s_iv_pct):.1f}%</b>'
            f'</div>', unsafe_allow_html=True)
        render_debug_panel(s_lvl)
        render_level_card(s_lvl)
        st.markdown("**ðŸ“‹ BackQuant Paste Block:**")
        st.code(backquant_block("US500", s_lvl, s["index_spot"], s["expiry"]), language=None)
        st.pyplot(gex_chart(s["filt"], s_lvl, s["index_spot"], "US500"))

    with tab2:
        override_tag = "  âš¡ override" if us100_override > 5000 else ""
        auto_note    = f"  auto: {q['auto_index']:.1f}" if us100_override > 5000 else ""
        st.markdown(f'<span class="source-badge badge-src">US100 {q["index_spot"]:.1f}{override_tag}  (QQQÃ—{q["scale"]:.4f}){auto_note}</span>',
                    unsafe_allow_html=True)
        iv_dir = "â†“" if q_iv_pct < 0 else "â†‘"
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;color:#4a6070;margin-bottom:8px;">'
            f'ATM IV: <b style="color:#ce93d8">{q["atm_iv"]*100:.1f}%</b> &nbsp;|&nbsp; '
            f'Open IV: <b style="color:#8a9bb0">{q_open_iv*100:.1f}%</b> &nbsp;|&nbsp; '
            f'IV Change: <b style="color:{"#ff5252" if q_iv_pct > 0 else "#00e676"}">{iv_dir}{abs(q_iv_pct):.1f}%</b>'
            f'</div>', unsafe_allow_html=True)
        render_debug_panel(q_lvl)
        render_level_card(q_lvl)
        st.markdown("**ðŸ“‹ BackQuant Paste Block:**")
        st.code(backquant_block("US100", q_lvl, q["index_spot"], q["expiry"]), language=None)
        st.pyplot(gex_chart(q["filt"], q_lvl, q["index_spot"], "US100"))

    st.markdown("""
    <div class="footer">
      Levels cached 30min Â· Regime engine live (ATM IV Â· session-open anchor)<br>
      Data is indicative â€” always verify with your broker before trading
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #3d5268;">
      <div style="font-size: 3rem; margin-bottom: 16px;">ðŸ“Š</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
        Set spot overrides if needed<br>then click Generate Levels
      </div>
    </div>
    """, unsafe_allow_html=True)

