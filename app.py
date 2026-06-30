"""
GEX Engine US500 / US100
Streamlit app. Gold removed yfinance SPY/QQQ only.
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

# PAGE CONFIG
st.set_page_config(
page_title="GEX Engine",
page_icon="GEX",
layout="centered",
initial_sidebar_state="collapsed",
)

# STYLING
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2upfamily=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;700;800&display=swap');

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
.lvl-red { color: #ff5252; }
.lvl-cyan { color: #00bcd4; }
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
.badge-src { background: #0d2a0d; color: #00e676; border: 1px solid #00e676; }

.regime-pos { color: #00e676; font-weight: 700; }
.regime-neg { color: #ff5252; font-weight: 700; }
.regime-blue { color: #4fc3f7; font-weight: 700; }
.regime-purple { color: #ce93d8; font-weight: 700; }
.regime-orange { color: #ffab40; font-weight: 700; }
.strength-bar {
 display: inline-block;
 width: 76px;
 height: 7px;
 background: #15253a;
 border-radius: 8px;
 overflow: hidden;
 vertical-align: middle;
 margin-left: 8px;
}
.strength-fill { display: block; height: 100%; border-radius: 8px; }

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

# CONSTANTS
RISK_FREE_RATE = 0.05
EQ_MULTIPLIER = 100

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
    ticker = yf.Ticker(symbol)
    etf_spot = first_valid_close(symbol)
    expiry = first_expiry(ticker, symbol)
    raw_df = option_chain_frame(ticker, expiry, symbol)

    has_override = override > override_min
    if not has_override:
        raise RuntimeError(
            f"Enter your live {display_label} CFD spot override so ETF option strikes convert to your broker price."
        )
    display_spot = float(override)
    conversion_source = "CFD_OVERRIDE"
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
        "spot": display_spot,
        "index_spot": display_spot,
        "auto_index": display_spot,
        "source_spot": etf_spot,
        "conversion_source": conversion_source,
        "atm_iv": atm_iv,
        "expiry": expiry,
        "scale": round(scale, 4),
        "block": block,
    }

# MATHS
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
 Dealers short calls long vanna buy spot when vol drops, sell when vol rises.
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
 Extract ATM implied volatility the strike closest to spot.
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
    piv = float(clean.loc[atm_idx, "put_iv"]) if "put_iv" in clean.columns else 0.0
    # Fall back to enriched columns if raw IV is zero.
    if civ <= 0.01 and "call_iv_c" in clean.columns:
        civ = float(clean.loc[atm_idx, "call_iv_c"])
    if piv <= 0.01 and "put_iv_c" in clean.columns:
        piv = float(clean.loc[atm_idx, "put_iv_c"])
    atm = (civ + piv) / 2 if (civ > 0.01 and piv > 0.01) else max(civ, piv)
    return atm if atm > 0.01 else 0.20

def store_session_open_iv(key, current_iv):
    """
 Store IV at session open NOT first refresh IV.
 Uses NY market open time (09:30 ET) as the reference point.
 If it is before market open or a new trading day, resets the stored IV.
 Returns (open_iv, iv_change, iv_change_pct).
 """
    now_ny = market_now("America/New_York")
    today_str = now_ny.strftime("%Y-%m-%d")
    open_key = f"{key}_open_iv"
    date_key = f"{key}_open_date"
    hour_ny = now_ny.hour + now_ny.minute / 60

    stored_date = st.session_state.get(date_key, "")
    stored_iv = st.session_state.get(open_key, None)

    # Reset if: new trading day OR before 09:30 (pre-market don't anchor to pre-market IV)
    is_new_day = stored_date != today_str
    is_premarket = hour_ny < 9.5
    is_afterhours = hour_ny >= 17.0

    if is_new_day or stored_iv is None:
        if not is_premarket and not is_afterhours:
            # Market is open: anchor to this IV as today's open reference.
            st.session_state[open_key] = current_iv
            st.session_state[date_key] = today_str
        stored_iv = current_iv

    open_iv = stored_iv
    iv_change = current_iv - open_iv
    iv_change_pct = (iv_change / open_iv * 100) if open_iv > 0 else 0.0
    return open_iv, iv_change, iv_change_pct


def dte(expiry_str):
    try:
        e = datetime.strptime(expiry_str[:10], "%Y-%m-%d").date()
        return max((e - date.today()).days, 1)
    except: return 20

# CORE COMPUTATION
def enrich(df, spot, multiplier, days=30):
    T = max(days / 365.0, 1/365)
    df = df.copy()
    civ_l, piv_l = [], []
    cg_l, pg_l = [], []
    cd_l, pd_l = [], []
    cv_l, pv_l = [], []  # vanna
    cc_l, pc_l = [], []  # charm

    for _, row in df.iterrows():
        K = row["strike"]
        civ = row["call_iv"] if row.get("call_iv", 0) > 0.01 else 0.15
        piv = row["put_iv"] if row.get("put_iv", 0) > 0.01 else (civ if civ > 0.01 else 0.15)
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
    df["call_gamma"] = cg_l; df["put_gamma"] = pg_l
    df["call_delta"] = cd_l; df["put_delta"] = pd_l
    df["call_vanna"] = cv_l; df["put_vanna"] = pv_l
    df["call_charm"] = cc_l; df["put_charm"] = pc_l

    # GEX: gamma OI multiplier S
    df["call_gex"] = df["call_gamma"] * df["call_oi"] * multiplier * spot**2
    df["put_gex"] = -df["put_gamma"] * df["put_oi"] * multiplier * spot**2
    df["net_gex"] = df["call_gex"] + df["put_gex"]

    # DEX: delta OI multiplier S
    df["call_dex"] = df["call_delta"] * df["call_oi"] * multiplier * spot
    df["put_dex"] = df["put_delta"] * df["put_oi"] * multiplier * spot
    df["net_dex"] = df["call_dex"] + df["put_dex"]

    # VEX (Vanna Exposure): vanna OI multiplier S
    df["call_vex"] = df["call_vanna"] * df["call_oi"] * multiplier * spot
    df["put_vex"] = -df["put_vanna"] * df["put_oi"] * multiplier * spot
    df["net_vex"] = df["call_vex"] + df["put_vex"]

    # CEX (Charm Exposure): per-day delta units from dealer perspective
    df["call_cex"] = -df["call_charm"] * df["call_oi"] * multiplier * spot / 365
    df["put_cex"] = df["put_charm"] * df["put_oi"] * multiplier * spot / 365
    df["net_cex"] = df["call_cex"] + df["put_cex"]

    return df

def _zero_cross_level(strikes, values):
    strikes = np.asarray(strikes, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(strikes) == 0:
        return 0.0
    crossings = np.where(np.diff(np.sign(values)))[0]
    if len(crossings) > 0:
        i = crossings[len(crossings) // 2]
        j = min(i + 1, len(strikes) - 1)
        v0, v1 = values[i], values[j]
        if v1 != v0:
            return float(strikes[i] - v0 * (strikes[j] - strikes[i]) / (v1 - v0))
    return float(strikes[np.argmin(np.abs(values))])


def assess_level_quality(filt, spot, level_values, oi_proxy=False):
    """Return a practical confidence label and warnings for the level set."""
    warnings_out = []
    unique_strikes = int(filt["strike"].nunique()) if filt is not None and not filt.empty else 0
    total_oi = float((filt["call_oi"] + filt["put_oi"]).sum()) if unique_strikes else 0.0
    strike_min = float(filt["strike"].min()) if unique_strikes else 0.0
    strike_max = float(filt["strike"].max()) if unique_strikes else 0.0
    span_pct = ((strike_max - strike_min) / spot * 100) if spot > 0 and unique_strikes else 0.0
    near_count = int(((filt["strike"] >= spot * 0.95) & (filt["strike"] <= spot * 1.05)).sum()) if unique_strikes else 0

    rounded_levels = [round(float(v), 1) for v in level_values.values()]
    level_counts = {v: rounded_levels.count(v) for v in set(rounded_levels)}
    max_level_cluster = max(level_counts.values()) if level_counts else 0
    clustered_levels = sorted([v for v, count in level_counts.items() if count >= 4])

    if oi_proxy:
        warnings_out.append(
            "Open interest was missing or zero; using equal-weight strike proxy to keep the app running."
        )
    if unique_strikes < 10:
        warnings_out.append(f"Only {unique_strikes} usable strikes after filtering.")
    if total_oi <= 0:
        warnings_out.append("No usable open interest in the selected strike window.")
    if span_pct < 6:
        warnings_out.append(f"Strike window is narrow ({span_pct:.1f}% of spot).")
    if near_count < 6:
        warnings_out.append(f"Only {near_count} strikes are within 5% of spot.")
    if max_level_cluster >= 5:
        warnings_out.append(
            "Many key levels collapsed to the same strike; treat regime and flips as low confidence."
        )

    if oi_proxy or total_oi <= 0 or unique_strikes < 8 or max_level_cluster >= 6:
        confidence = "LOW"
    elif unique_strikes < 15 or span_pct < 8 or max_level_cluster >= 4:
        confidence = "WATCH"
    else:
        confidence = "OK"

    if confidence == "OK":
        summary = f"OK - {unique_strikes} strikes, {span_pct:.1f}% span"
    elif confidence == "WATCH":
        summary = f"WATCH - verify clustered levels ({unique_strikes} strikes, {span_pct:.1f}% span)"
    else:
        summary = f"LOW - verify chain quality before trading ({unique_strikes} strikes, {span_pct:.1f}% span)"

    return {
        "confidence": confidence,
        "summary": summary,
        "warnings": warnings_out,
        "unique_strikes": unique_strikes,
        "total_oi": round(total_oi, 0),
        "span_pct": round(span_pct, 2),
        "near_count": near_count,
        "max_level_cluster": max_level_cluster,
        "clustered_levels": clustered_levels,
    }


def classify_market_state(gex_positive, dex_positive, vanna_bullish, vanna_bearish, vanna_fuel):
    """GEX sets regime, DEX direction, Vanna only adds acceleration risk."""
    if not gex_positive and dex_positive and vanna_fuel >= 70:
        return "SQUEEZE RISK", "Dealers may chase upside; breakout continuation favored", "regime-orange"
    if not gex_positive and not dex_positive:
        return "VOLATILE BEARISH", "Trend expansion with downside pressure; rallies may fail", "regime-neg"
    if gex_positive and dex_positive:
        return "CONTROLLED BULLISH", "Bullish drift possible, but upside likely capped", "regime-blue"
    if gex_positive and not dex_positive:
        return "PINNED RANGE", "Mean reversion dominates; fade extremes first", "regime-blue"
    if vanna_bearish:
        return "DOWNWARD ACCELERATION RISK", "Vanna warns downside may speed up", "regime-neg"
    if vanna_bullish:
        return "UPSIDE ACCELERATION RISK", "Vanna fuel supports upside continuation", "regime-orange"
    return "NEUTRAL FLOW", "No clean dealer-flow edge", "regime-purple"


def build_flow_confidence(gex_positive, dex_positive, vanna_bullish, vanna_bearish,
                          vanna_fuel, cex_positive, days, exposure_stats,
                          data_quality):
    """Create one tactical read without letting lower-weight flows override GEX."""
    def clamp(v, lo=5, hi=95):
        return int(max(lo, min(hi, round(v))))

    gex_balance = exposure_stats.get("gex_balance", 0.0)
    dex_balance = exposure_stats.get("dex_balance", 0.0)
    cex_balance = exposure_stats.get("cex_balance", 0.0)
    gex_distance = exposure_stats.get("gex_distance_pct", 0.0)

    gex_score = clamp(45 + 30 * min(gex_distance / 1.0, 1.0) + 20 * gex_balance)
    dex_score = clamp(35 + 55 * dex_balance)
    charm_weight = 0.35 if days >= 21 else 1.0
    charm_score = clamp((30 + 60 * cex_balance) * charm_weight)
    vanna_score = clamp(vanna_fuel if (vanna_bullish or vanna_bearish) else max(20, vanna_fuel))
    overall_score = clamp(
        gex_score * 0.40 + dex_score * 0.35 + vanna_score * 0.15 + charm_score * 0.10
    )
    if overall_score >= 70:
        flow_quality = "OK"
    elif overall_score >= 50:
        flow_quality = "WATCH"
    else:
        flow_quality = "LOW"
    flow_summary = f"{flow_quality} - weighted flow confidence {overall_score}% (GEX/DEX led)"

    gex_name = "POSITIVE GAMMA" if gex_positive else "NEGATIVE GAMMA"
    gex_note_1 = "Mean reversion dominant" if gex_positive else "Expansion risk dominant"
    gex_note_2 = "Breakout probability suppressed" if gex_positive else "Breakouts can extend faster"

    dex_name = "BULLISH DELTA" if dex_positive else "BEARISH DELTA"
    dex_note_1 = "Upside pressure exists" if dex_positive else "Downside pressure exists"
    dex_note_2 = "Expansion capped by gamma" if gex_positive else "Expansion enabled by gamma"

    if vanna_bullish:
        vanna_name = "VANNA FUEL"
        vanna_note_1 = "Squeeze support active"
        vanna_note_2 = "Vol crush favorable" if gex_positive else "Can accelerate with trend"
    elif vanna_bearish:
        vanna_name = "VANNA DRAG"
        vanna_note_1 = "Downside acceleration risk"
        vanna_note_2 = "Respect failed supports" if not gex_positive else "Gamma may slow expansion"
    else:
        vanna_name = "VANNA NEUTRAL"
        vanna_note_1 = "Acceleration fuel limited"
        vanna_note_2 = "Do not force momentum reads"

    charm_name = "CHARM DRIFT"
    charm_note_1 = "Small bullish close bias" if cex_positive else "Small bearish close bias"
    charm_note_2 = "Low weight due to DTE" if days >= 21 else "Useful for timing only"

    if gex_positive and not dex_positive:
        state = "PINNED RANGE"
        tactic = "Fade extremes first; sell failed rallies near resistance; avoid aggressive breakout trades; expect pinning near HVL / Max Pain."
    elif gex_positive and dex_positive:
        state = "CONTROLLED BULLISH"
        tactic = "Buy pullbacks into support; take profits into call walls; avoid chasing extensions while gamma is pinning."
    elif not gex_positive and dex_positive:
        state = "SQUEEZE RISK" if vanna_bullish and vanna_score >= 60 else "BULLISH EXPANSION WATCH"
        tactic = "Favor momentum only after reclaimed resistance; use DEX confirmation; avoid fading a clean breakout."
    elif not gex_positive and not dex_positive:
        state = "BEARISH EXPANSION"
        tactic = "Sell failed rallies; respect downside breaks; avoid mean-reversion longs until price reclaims key gamma levels."
    else:
        state = "MIXED FLOW"
        tactic = "Wait for price to confirm at HVL, walls, or flips before sizing; keep trades tactical."

    if data_quality != "OK":
        tactic = f"{tactic} Chain quality is {data_quality}; reduce size until levels confirm."

    hierarchy = [
        {"name": gex_name, "score": gex_score, "note_1": gex_note_1, "note_2": gex_note_2},
        {"name": dex_name, "score": dex_score, "note_1": dex_note_1, "note_2": dex_note_2},
        {"name": vanna_name, "score": vanna_score, "note_1": vanna_note_1, "note_2": vanna_note_2},
        {"name": charm_name, "score": charm_score, "note_1": charm_note_1, "note_2": charm_note_2},
    ]

    return {
        "state": state,
        "hierarchy": hierarchy,
        "tactical": tactic,
        "overall_score": overall_score,
        "quality": flow_quality,
        "summary": flow_summary,
    }


def score_level_strength(by_strike, spot, level_map, gex_positive, dex_positive):
    """Estimate whether each major level is likely to hold or break."""
    exposure_cols = ["net_gex", "net_dex", "net_vex", "net_cex"]
    max_exposure = 1.0
    for col in exposure_cols:
        if col in by_strike:
            max_exposure = max(max_exposure, float(by_strike[col].abs().max()))

    def nearest_row(level):
        idx = int(np.argmin(np.abs(by_strike.index.to_numpy(dtype=float) - float(level))))
        return by_strike.iloc[idx], float(by_strike.index[idx])

    scored = []
    levels = {k: float(v) for k, v in level_map.items()}
    for name, level in levels.items():
        row, actual = nearest_row(level)
        abs_pressure = sum(abs(float(row.get(col, 0.0))) for col in exposure_cols)
        pressure_score = min(45.0, 45.0 * abs_pressure / (max_exposure * len(exposure_cols)))
        distance_pct = abs(actual - spot) / max(spot, 1.0)
        distance_score = max(0.0, 20.0 - min(20.0, distance_pct * 500.0))
        cluster = sum(abs(other - level) <= max(spot * 0.0025, 5.0) for other in levels.values())
        confluence_score = min(20.0, cluster * 5.0)

        is_resistance = "Call" in name or actual > spot
        is_support = "Put" in name or actual < spot
        flow_score = 0.0
        if gex_positive:
            flow_score += 10.0
        if is_support and dex_positive:
            flow_score += 10.0
        if is_resistance and not dex_positive:
            flow_score += 10.0

        score = int(round(min(100.0, pressure_score + distance_score + confluence_score + flow_score)))
        if score >= 70:
            verdict, cls = "HOLD LIKELY", "regime-blue"
        elif score >= 45:
            verdict, cls = "TEST / RESPECT", "regime-orange"
        else:
            verdict, cls = "BREAK RISK", "regime-neg"
        scored.append({
            "name": name,
            "level": round(level, 1),
            "score": score,
            "verdict": verdict,
            "class": cls,
        })
    return scored


def build_intraday_gamma_nodes(by_strike, spot, call_wall, put_wall, gex_positive, dex_positive):
    """Rank local gamma shelves for intraday trading, not just largest raw walls."""
    strikes = by_strike.index.to_numpy(dtype=float)
    if len(strikes) == 0 or spot <= 0:
        return {"call": [], "put": []}

    spacing = float(np.median(np.diff(np.sort(strikes)))) if len(strikes) > 1 else max(spot * 0.002, 5.0)
    exclude_band = max(spacing * 0.75, spot * 0.0008)
    cluster_band = max(spacing * 1.5, spot * 0.0025)

    def side_nodes(side):
        if side == "call":
            exposure = by_strike["call_gex"].clip(lower=0).abs()
            oi = by_strike["call_oi"].abs()
            candidates = exposure[exposure.index >= spot]
            primary = call_wall
            is_support = False
        else:
            exposure = by_strike["put_gex"].abs()
            oi = by_strike["put_oi"].abs()
            candidates = exposure[exposure.index <= spot]
            primary = put_wall
            is_support = True
        if candidates.empty:
            candidates = exposure

        max_exp = max(float(exposure.max()), 1.0)
        max_oi = max(float(oi.max()), 1.0)
        nodes = []
        for strike, raw in candidates.items():
            strike = float(strike)
            if abs(strike - float(primary)) <= exclude_band:
                continue
            mag_score = float(abs(raw)) / max_exp
            distance_pct = abs(strike - spot) / max(spot, 1.0)
            distance_score = float(np.exp(-distance_pct / 0.018))
            nearby = exposure[(exposure.index >= strike - cluster_band) & (exposure.index <= strike + cluster_band)]
            cluster_score = min(1.0, float(nearby.sum()) / max_exp)
            oi_score = float(oi.get(strike, 0.0)) / max_oi
            importance = 100.0 * (
                0.46 * mag_score + 0.30 * distance_score + 0.14 * cluster_score + 0.10 * oi_score
            )

            flow_bonus = 8.0 if gex_positive else 0.0
            if is_support and dex_positive:
                flow_bonus += 8.0
            if (not is_support) and (not dex_positive):
                flow_bonus += 8.0
            hold_score = int(round(min(100.0, importance * 0.82 + flow_bonus)))
            if hold_score >= 70:
                verdict, cls = "HOLD LIKELY", "regime-blue"
            elif hold_score >= 50:
                verdict, cls = "TRADE ZONE", "regime-orange"
            else:
                verdict, cls = "WEAK / BREAK RISK", "regime-neg"
            trade_zone = "YES" if hold_score >= 55 and distance_pct <= 0.035 else "WATCH"
            nodes.append({
                "name": "Secondary Call Node" if side == "call" else "Secondary Put Node",
                "level": round(strike, 1),
                "score": hold_score,
                "importance": int(round(importance)),
                "verdict": verdict,
                "trade_zone": trade_zone,
                "distance_pct": round(distance_pct * 100, 2),
                "class": cls,
            })
        nodes.sort(key=lambda n: (n["score"], -n["distance_pct"]), reverse=True)
        return nodes[:3]

    return {"call": side_nodes("call"), "put": side_nodes("put")}


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

    oi_proxy = False
    near_spot_mask = (df["strike"] >= spot * (1 - filter_pct)) & (df["strike"] <= spot * (1 + filter_pct))
    real_oi_mask = (df["call_oi"] > 0) | (df["put_oi"] > 0)
    if real_oi_mask.any() and (near_spot_mask & real_oi_mask).sum() >= 6:
        df = df[real_oi_mask].copy()
    else:
        oi_proxy = True
        df = df.copy()
        spacing = float(df["strike"].sort_values().diff().median())
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = max(spot * 0.002, 5.0)
        distance = (df["strike"] - spot).abs() / max(spacing, 1.0)
        local_weight = np.exp(-0.5 * (distance / 6.0) ** 2)
        df["call_oi"] = np.where(df["call_oi"] > 0, df["call_oi"], np.maximum(1.0, 100.0 * local_weight))
        df["put_oi"] = np.where(df["put_oi"] > 0, df["put_oi"], np.maximum(1.0, 100.0 * local_weight))
        df = enrich(df, spot, EQ_MULTIPLIER, days)
    df = df.sort_values("strike").reset_index(drop=True)

    filt = None
    for pct in sorted(set([filter_pct, filter_pct + 0.05, 0.30, 0.50, 1.0])):
        mask = (df["strike"] >= spot * (1 - pct)) & (df["strike"] <= spot * (1 + pct))
        candidate = df[mask].sort_values("strike").reset_index(drop=True)
        if len(candidate) >= 10:
            filt = candidate
            break
    if filt is None or filt.empty:
        filt = df.sort_values("strike").reset_index(drop=True)

    by_strike = filt.groupby("strike", sort=True).sum(numeric_only=True)
    strikes = by_strike.index.to_numpy(dtype=float)

    net_gex = by_strike["net_gex"]
    call_gex = by_strike["call_gex"]
    put_gex = by_strike["put_gex"]
    hvl = float(net_gex.abs().idxmax())
    call_above = call_gex[call_gex.index >= spot]
    put_below = put_gex[put_gex.index <= spot]
    call_wall = float(call_above.idxmax()) if not call_above.empty else float(call_gex.idxmax())
    put_wall = float(put_below.idxmin()) if not put_below.empty else float(put_gex.idxmin())
    gflip = _zero_cross_level(strikes, by_strike["net_gex"].cumsum().to_numpy())
    gflip = float(np.clip(gflip, min(put_wall, call_wall), max(put_wall, call_wall)))

    call_dex = by_strike["call_dex"]
    put_dex = by_strike["put_dex"]
    call_dex_above = call_dex[call_dex.index >= spot]
    put_dex_below = put_dex[put_dex.index <= spot]
    call_delta_wall = float(call_dex_above.idxmax()) if not call_dex_above.empty else float(call_dex.idxmax())
    put_delta_wall = float(put_dex_below.idxmin()) if not put_dex_below.empty else float(put_dex.idxmin())
    dflip = _zero_cross_level(strikes, by_strike["net_dex"].cumsum().to_numpy())
    dflip = float(np.clip(dflip, min(put_delta_wall, call_delta_wall), max(put_delta_wall, call_delta_wall)))

    vanna_flip = _zero_cross_level(strikes, by_strike["net_vex"].cumsum().to_numpy())

    sa = filt["strike"].to_numpy(dtype=float)
    coi = filt["call_oi"].to_numpy(dtype=float)
    poi = filt["put_oi"].to_numpy(dtype=float)
    losses = [np.sum(np.maximum(sa - s, 0) * coi) + np.sum(np.maximum(s - sa, 0) * poi) for s in sa]
    max_pain = float(sa[int(np.argmin(losses))])

    total_net_gex = float(filt["net_gex"].sum())
    total_net_dex = float(filt["net_dex"].sum())
    total_net_vex = float(filt["net_vex"].sum())
    total_net_cex = float(filt["net_cex"].sum())

    quality = assess_level_quality(filt, spot, {
        "HVL": hvl,
        "Call Wall": call_wall,
        "Put Wall": put_wall,
        "GEX Flip": gflip,
        "Call DEX Wall": call_delta_wall,
        "Put DEX Wall": put_delta_wall,
        "DEX Flip": dflip,
        "Vanna Flip": vanna_flip,
        "Max Pain": max_pain,
    }, oi_proxy=oi_proxy)

    gex_positive = spot > gflip
    dex_positive = total_net_dex > 0
    vex_positive = total_net_vex > 0
    cex_positive = total_net_cex > 0

    regime = "POSITIVE GAMMA" if gex_positive else "NEGATIVE GAMMA"
    dex_bias = "BULLISH UP" if dex_positive else "BEARISH DOWN"
    intraday_nodes = build_intraday_gamma_nodes(
        by_strike, spot, call_wall, put_wall, gex_positive, dex_positive
    )
    secondary_call_wall = (
        intraday_nodes["call"][0]["level"] if intraday_nodes["call"] else round(call_wall, 1)
    )
    secondary_put_wall = (
        intraday_nodes["put"][0]["level"] if intraday_nodes["put"] else round(put_wall, 1)
    )

    price_up = price_change >= 0
    iv_falling = iv_change <= 0
    if not has_open_price:
        if not gex_positive and not vex_positive:
            vanna_regime = "HIGH SQUEEZE FUEL"
            vanna_note = "VANNA FUEL HIGH - acceleration risk elevated"
        elif not gex_positive:
            vanna_regime = "MODERATE VANNA FUEL"
            vanna_note = "VANNA FUEL MODERATE - trend acceleration possible"
        else:
            vanna_regime = "LOW ACCELERATION RISK"
            vanna_note = "VANNA FUEL LOW - gamma pinning reduces acceleration"
    elif price_up and iv_falling:
        if not vex_positive:
            vanna_regime = "VOL-CRUSH BULLISH"
            vanna_note = "PRICE UP + IV DOWN + VEX NEGATIVE = SQUEEZE FUEL"
        else:
            vanna_regime = "VOL-CRUSH BEARISH"
            vanna_note = "PRICE UP + IV DOWN + VEX POSITIVE = RALLY FADING"
    elif (not price_up) and (not iv_falling):
        if vex_positive:
            vanna_regime = "VOL-SPIKE BULLISH"
            vanna_note = "PRICE DOWN + IV UP + VEX POSITIVE = FLOOR POSSIBLE"
        else:
            vanna_regime = "VOL-SPIKE BEARISH"
            vanna_note = "PRICE DOWN + IV UP + VEX NEGATIVE = CRASH ACCELERATOR"
    elif price_up and not iv_falling:
        vanna_regime = "VOL-SPIKE UNSTABLE"
        vanna_note = "PRICE UP + IV UP - UNSTABLE; WATCH REVERSAL"
    else:
        vanna_regime = "VOL-CRUSH NEUTRAL"
        vanna_note = "PRICE DOWN + IV DOWN - RANGE / EXHAUSTION; COIL FORMING"

    charm_flow = "BULLISH BLEED" if cex_positive else "BEARISH BLEED"
    charm_note = "AFTERNOON MELT-UP BIAS" if cex_positive else "AFTERNOON FADE BIAS"
    if days >= 21:
        charm_note = f"CHARM: {'up' if cex_positive else 'down'} (low weight - {days}+ DTE)"

    vanna_bullish = "BULLISH" in vanna_regime or "SQUEEZE FUEL" in vanna_note
    vanna_bearish = ("BEARISH" in vanna_regime or "CRASH ACCELERATOR" in vanna_note) and not vanna_bullish
    vanna_neutral = not vanna_bullish and not vanna_bearish
    vanna_fuel = 35
    if (not gex_positive) and vanna_bullish:
        vanna_fuel = 85
    elif (not gex_positive) and vanna_bearish:
        vanna_fuel = 80
    elif vanna_bullish or vanna_bearish:
        vanna_fuel = 60
    elif gex_positive:
        vanna_fuel = 25

    price_up_strict = price_change > 0
    iv_falling_strict = iv_change < 0
    price_dn_strict = price_change < 0
    iv_rising_strict = iv_change > 0
    no_live_input = (not has_open_price and iv_change == 0.0)
    bull_signals = 0 if no_live_input else sum([price_up_strict, iv_falling_strict, vanna_bullish])
    bear_signals = 0 if no_live_input else sum([price_dn_strict, iv_rising_strict, vanna_bearish])

    gex_char = "pinning" if gex_positive else "expansion"
    if bull_signals == 3:
        intraday_condition = "BULLISH SQUEEZE" if gex_positive else "SQUEEZE TREND UP"
        strategy = f"Long bias - IV crush supportive - GEX {gex_char}"
        condition_cls = "regime-pos" if gex_positive else "regime-orange"
    elif bear_signals == 3:
        intraday_condition = "BEARISH SQUEEZE" if gex_positive else "CRASH / TREND DOWN"
        strategy = f"Short bias - IV spike pressuring - GEX {gex_char}"
        condition_cls = "regime-neg"
    elif bull_signals == 2:
        intraday_condition = "BULLISH LEAN" if dex_positive else "UNSTABLE BULLISH"
        strategy = f"Long bias - flow supportive - GEX {gex_char} - confirm price action"
        condition_cls = "regime-pos" if dex_positive else "regime-orange"
    elif bear_signals == 2:
        intraday_condition = "BEARISH LEAN" if not dex_positive else "UNSTABLE BEARISH"
        strategy = f"Short bias - flow pressuring - GEX {gex_char} - confirm breakdown"
        condition_cls = "regime-neg" if not dex_positive else "regime-orange"
    elif gex_positive and dex_positive:
        intraday_condition = "STABLE BULLISH"
        strategy = "Buy pullbacks - fade resistance - gamma pinning; avoid chasing"
        condition_cls = "regime-pos"
    elif gex_positive and not dex_positive:
        intraday_condition = "STABLE BEARISH"
        strategy = "Sell rallies - fade bounces - gamma pinning; avoid breakout longs"
        condition_cls = "regime-neg"
    elif not gex_positive and dex_positive:
        intraday_condition = "SQUEEZE WATCH"
        strategy = "Momentum long watch - gamma expanding - confirm IV direction"
        condition_cls = "regime-orange"
    else:
        intraday_condition = "TREND / RANGE DOWN"
        strategy = "Short bias - gamma expanding - no clear squeeze setup"
        condition_cls = "regime-neg"

    if bull_signals == 3 and dex_positive:
        entry_trigger = "LONG: price up + IV down + Vanna bullish + DEX confirmed - squeeze active"
        trigger_cls = "regime-pos" if gex_positive else "regime-orange"
    elif bull_signals == 3:
        entry_trigger = "SQUEEZE LONG: price up + IV down + Vanna bullish - DEX lagging; wait for flip"
        trigger_cls = "regime-orange"
    elif bear_signals == 3 and not dex_positive:
        entry_trigger = "SHORT: price down + IV up + Vanna bearish + DEX confirmed - trend active"
        trigger_cls = "regime-neg"
    elif bear_signals == 3:
        entry_trigger = "BREAKDOWN RISK: price down + IV up + Vanna bearish - DEX still holding"
        trigger_cls = "regime-neg"
    elif bull_signals == 2 and dex_positive and iv_falling_strict:
        entry_trigger = "LONG LEAN: majority bullish flow + DEX confirmed - watch IV continuation"
        trigger_cls = "regime-pos"
    elif bear_signals == 2 and not dex_positive and iv_rising_strict:
        entry_trigger = "SHORT LEAN: majority bearish flow + DEX confirmed - watch IV continuation"
        trigger_cls = "regime-neg"
    elif not has_open_price:
        entry_trigger = "WAITING - enter live spot and open price to unlock triggers"
        trigger_cls = "regime-purple"
    else:
        entry_trigger = "MIXED SIGNALS - 2 of 3 flow signals needed for entry"
        trigger_cls = "regime-purple"

    market_state, market_read, market_cls = classify_market_state(
        gex_positive, dex_positive, vanna_bullish, vanna_bearish, vanna_fuel
    )
    level_strength = score_level_strength(by_strike, spot, {
        "HVL": hvl,
        "GEX Flip": gflip,
        "Call Wall": call_wall,
        "Put Wall": put_wall,
        "Call DEX Wall": call_delta_wall,
        "Put DEX Wall": put_delta_wall,
        "Max Pain": max_pain,
    }, gex_positive, dex_positive)

    def exposure_balance(col, total):
        gross = float(filt[col].abs().sum()) if col in filt else 0.0
        return 0.0 if gross <= 0 else min(1.0, abs(float(total)) / gross)

    flow_confidence = build_flow_confidence(
        gex_positive,
        dex_positive,
        vanna_bullish,
        vanna_bearish,
        vanna_fuel,
        cex_positive,
        days,
        {
            "gex_balance": exposure_balance("net_gex", total_net_gex),
            "dex_balance": exposure_balance("net_dex", total_net_dex),
            "cex_balance": exposure_balance("net_cex", total_net_cex),
            "gex_distance_pct": abs(spot - gflip) / max(spot, 1.0) * 100.0,
        },
        quality["confidence"],
    )

    return {
        "HVL": round(hvl, 1),
        "Call Wall": round(call_wall, 1),
        "Put Wall": round(put_wall, 1),
        "Secondary Call Wall": round(float(secondary_call_wall), 1),
        "Secondary Put Wall": round(float(secondary_put_wall), 1),
        "Intraday Gamma Nodes": intraday_nodes,
        "GEX Flip": round(gflip, 1),
        "Call DEX Wall": round(call_delta_wall, 1),
        "Put DEX Wall": round(put_delta_wall, 1),
        "DEX Flip": round(dflip, 1),
        "Vanna Flip": round(vanna_flip, 1),
        "Max Pain": round(max_pain, 1),
        "Net GEX $B": round(total_net_gex / 1e9, 3),
        "Net DEX $B": round(total_net_dex / 1e9, 3),
        "Net VEX $M": round(total_net_vex / 1e6, 2),
        "Net CEX (daily)": round(total_net_cex, 0),
        "Regime": regime,
        "DEX Bias": dex_bias,
        "Vanna Regime": vanna_regime,
        "Charm Flow": charm_flow,
        "Market State": market_state,
        "Market Read": market_read,
        "Market Class": market_cls,
        "Vanna Fuel": vanna_fuel,
        "Level Strength": level_strength,
        "Flow Confidence": flow_confidence,
        "Intraday Condition": intraday_condition,
        "Strategy": strategy,
        "Condition Class": condition_cls,
        "Vanna Note": vanna_note,
        "Charm Note": charm_note,
        "Entry Trigger": entry_trigger,
        "Trigger Class": trigger_cls,
        "IV Change": round(iv_change * 100, 2),
        "Price Change": round(price_change, 2),
        "Data Quality": flow_confidence["quality"],
        "Quality Summary": flow_confidence["summary"],
        "Chain Quality": quality["confidence"],
        "Chain Quality Summary": quality["summary"],
        "Quality Warnings": quality["warnings"],
        "_debug": {
            "price_change": round(price_change, 4),
            "iv_change": round(iv_change * 100, 4),
            "price_up_strict": price_up_strict,
            "iv_falling_strict": iv_falling_strict,
            "vanna_bullish": vanna_bullish,
            "vanna_bearish": vanna_bearish,
            "vanna_neutral": vanna_neutral,
            "bull_signals": bull_signals,
            "bear_signals": bear_signals,
            "no_live_input": no_live_input,
            "gex_positive": gex_positive,
            "dex_positive": dex_positive,
            "vex_positive": vex_positive,
            "cex_positive": cex_positive,
            "unique_strikes": quality["unique_strikes"],
            "total_oi": quality["total_oi"],
            "span_pct": quality["span_pct"],
            "near_count": quality["near_count"],
            "max_level_cluster": quality["max_level_cluster"],
            "clustered_levels": quality["clustered_levels"],
            "oi_proxy": oi_proxy,
        },
    }, filt

def expected_move(spot, net_b, d=20):
    # IV proxy: start at 18% for gold (lower than 30%), shrink slightly with positive GEX
    # Positive GEX = pinning regime = lower realised vol
    d = max(int(d), 1)
    iv = max(0.10, 0.18 - abs(net_b) * 0.01)
    em = spot * iv * (d / 365.0) ** 0.5
    return round(spot - em, 1), round(spot + em, 1)

def backquant_block(label, lvl, spot, expiry):
    now = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    d = dte(expiry)
    lo, hi = expected_move(spot, lvl["Net GEX $B"], d)
    flow = lvl.get("Flow Confidence", {})
    chain_text = lvl.get("Chain Quality Summary", "")
    chain_text = f"Chain Quality: {chain_text}\n" if chain_text else ""
    warnings_text = "\n".join(f"Quality Warning: {w}" for w in lvl.get("Quality Warnings", []))
    warnings_text = f"{warnings_text}\n" if warnings_text else ""
    hierarchy_lines = "\n".join(
        f"{idx}. {item['name']} [{item['score']}%]\n"
        f"   -> {item['note_1']}\n"
        f"   -> {item['note_2']}"
        for idx, item in enumerate(flow.get("hierarchy", []), start=1)
    )
    if hierarchy_lines:
        hierarchy_lines = f"Flow Hierarchy:\n{hierarchy_lines}\n"
    strength_lines = "\n".join(
        f"{item['name']}: ${item['level']:,.1f} - {item['verdict']} ({item['score']}%)"
        for item in lvl.get("Level Strength", [])
    )
    intraday_lines = []
    for side_name, nodes in [
        ("Secondary Resistance Nodes", lvl.get("Intraday Gamma Nodes", {}).get("call", [])),
        ("Secondary Support Nodes", lvl.get("Intraday Gamma Nodes", {}).get("put", [])),
    ]:
        if nodes:
            intraday_lines.append(side_name)
            intraday_lines.extend(
                f"- ${n['level']:,.1f} | Hold {n['score']}% | {n['verdict']} | Trade Zone: {n['trade_zone']}"
                for n in nodes
            )
    intraday_text = "\n".join(intraday_lines)
    intraday_text = f"{intraday_text}\n" if intraday_text else ""
    return (
f"GEX + DEX + VANNA + CHARM Levels [{label}] - {now}\n"
f"Data Quality: {lvl.get('Quality Summary', 'n/a')}\n"
f"{chain_text}"
f"{warnings_text}"
f"Market State: {flow.get('state', lvl['Market State'])} | {lvl['Market Read']}\n"
f"{hierarchy_lines}"
f"Regime: {lvl['Regime']} | Direction: {lvl['DEX Bias']}\n"
f"Vanna: {lvl['Vanna Note']} | Charm: {lvl['Charm Note']}\n"
f"Tactical Execution: {flow.get('tactical', lvl['Strategy'])}\n"
f"---\n"
f"Level Strength / Hold Risk:\n"
f"{strength_lines}\n"
f"---\n"
f"Core GEX Levels:\n"
f"HVL (Gamma Anchor): ${lvl['HVL']:,.1f}\n"
f"GEX Flip: ${lvl['GEX Flip']:,.1f}\n"
f"Call Resistance: ${lvl['Call Wall']:,.1f}\n"
f"Put Support: ${lvl['Put Wall']:,.1f}\n"
f"Secondary Intraday GEX Nodes:\n"
f"{intraday_text}"
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

# CHART
def gex_chart(filt, lvl, spot, label):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#080c10")
    ax.set_facecolor("#080c10")
    x = filt["strike"]
    bw = (x.max()-x.min()) / max(len(x),1) * 0.75
    ax.bar(x, filt["call_gex"]/1e9, width=bw, color="#00e676", alpha=0.7, label="Call GEX")
    ax.bar(x, filt["put_gex"] /1e9, width=bw, color="#ff5252", alpha=0.7, label="Put GEX")
    ax.axhline(0, color="#2a3a2a", linewidth=0.8)
    for val, col, ls, lw, lbl in [
    (lvl["HVL"], "#ffd740", "-", 2.0, f"HVL {lvl['HVL']}"),
    (lvl["Call Wall"], "#00e676", "-", 2.0, f"CALL WALL {lvl['Call Wall']}"),
    (lvl["Put Wall"], "#ff5252", "-", 2.0, f"PUT WALL {lvl['Put Wall']}"),
    (lvl["GEX Flip"], "#00bcd4", "--", 1.5, f"GEX FLIP {lvl['GEX Flip']}"),
    (lvl["Vanna Flip"], "#ce93d8", "-.", 1.5, f"VANNA FLIP {lvl['Vanna Flip']}"),
    (lvl["Max Pain"], "#ffab40", ":", 1.2, f"MAX PAIN {lvl['Max Pain']}"),
    (spot, "#ffffff", "--", 1.0, f"SPOT {spot:.1f}"),
    ]:
        ax.axvline(val, color=col, linestyle=ls, linewidth=lw, label=lbl)

    for side, col, prefix in [
        ("call", "#66ffa6", "2ND CALL"),
        ("put", "#ff8a80", "2ND PUT"),
    ]:
        for idx, node in enumerate(lvl.get("Intraday Gamma Nodes", {}).get(side, []), start=1):
            ax.axvline(
                node["level"],
                color=col,
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
                label=f"{prefix} {idx} {node['level']}",
            )

    ax.set_title(
    f"{label} | {datetime.now().strftime('%d %b %Y')} | "
    f"Regime: {lvl['Regime']} | Net GEX ${lvl['Net GEX $B']}B",
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

        # FETCH ALL DATA
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all(_cache_buster: int = 0,
us500_override: float = 0.0,
us100_override: float = 0.0):
    results = {}
    if us500_override > 1000:
        results["spy"] = build_scaled_index(
        "SPY", "^GSPC", fallback_scale=10.0, override=us500_override,
        override_min=1000.0, display_label="US500",
        )
    if us100_override > 5000:
        results["qqq"] = build_scaled_index(
        "QQQ", "^NDX", fallback_scale=40.0, override=us100_override,
        override_min=5000.0, display_label="US100",
        )
    if not results:
        raise RuntimeError("Enter either a US500 CFD spot or a US100 CFD spot.")
    return results
    # -- LEVEL CARD RENDERER -------------------------------------------------------
def render_level_card(lvl):
    reg_cls = "regime-blue" if "POSITIVE" in lvl["Regime"] else "regime-neg"
    dex_cls = "regime-pos" if "BULLISH" in lvl["DEX Bias"] else "regime-neg"
    van_cls = "regime-neg" if "BEARISH" in lvl["Vanna Regime"] else "regime-purple"
    chm_cls = "regime-orange" if "BULLISH" in lvl["Charm Flow"] else "regime-neg"
    cond_cls = lvl["Condition Class"]
    quality_cls = {
        "OK": "regime-pos",
        "WATCH": "regime-orange",
        "LOW": "regime-neg",
    }.get(lvl.get("Data Quality"), "regime-purple")
    chain_quality_cls = {
        "OK": "regime-pos",
        "WATCH": "regime-orange",
        "LOW": "regime-neg",
    }.get(lvl.get("Chain Quality"), "regime-purple")
    warnings_html = "".join(
        f"<div class=\"level-row\" style=\"padding:3px 0;\"><span class=\"level-label\">Chain Warning</span>"
        f"<span class=\"regime-orange\" style=\"font-size:0.76rem;text-align:right;max-width:72%;\">{escape(w)}</span></div>"
        for w in lvl.get("Quality Warnings", [])
    )

    cex_val = lvl["Net CEX (daily)"]
    cex_str = f"+{cex_val:,.0f}" if cex_val >= 0 else f"{cex_val:,.0f}"
    strength_html = "".join(
        f"""<div class="level-row">
 <span class="level-label">{escape(item['name'])} {item['level']:,.1f}</span>
 <span class="{item['class']}" style="font-size:0.78rem;text-align:right;">
 {item['verdict']} {item['score']}%
 <span class="strength-bar"><span class="strength-fill" style="width:{item['score']}%;background:{'#4fc3f7' if item['score'] >= 70 else ('#ffab40' if item['score'] >= 45 else '#ff5252')};"></span></span>
 </span></div>"""
        for item in lvl.get("Level Strength", [])
    )
    node_rows = []
    for label, nodes in [
        ("Resistance", lvl.get("Intraday Gamma Nodes", {}).get("call", [])),
        ("Support", lvl.get("Intraday Gamma Nodes", {}).get("put", [])),
    ]:
        for item in nodes:
            node_rows.append(
                f"""<div class="level-row">
 <span class="level-label">{label} {item['level']:,.1f}</span>
 <span class="{item['class']}" style="font-size:0.78rem;text-align:right;">
 Hold {item['score']}% - {item['verdict']} - Zone {item['trade_zone']}
 </span></div>"""
            )
    intraday_html = "".join(node_rows)
    flow = lvl.get("Flow Confidence", {})
    flow_rows = "".join(
        f"""<div class="level-row" style="align-items:flex-start;padding:5px 0;">
 <span class="level-label">{idx}. {escape(item['name'])} [{item['score']}%]</span>
 <span class="level-val" style="font-size:0.76rem;color:#8a9bb0;text-align:right;max-width:62%;">
 {escape(item['note_1'])}<br>{escape(item['note_2'])}
 </span></div>"""
        for idx, item in enumerate(flow.get("hierarchy", []), start=1)
    )
    flow_state = escape(flow.get("state", lvl.get("Market State", "n/a")))
    flow_tactical = escape(flow.get("tactical", lvl.get("Strategy", "n/a")))

    st.markdown(f"""
 <div class="level-card">

  <div class="section-divider"></div>
  <div class="section-header">FLOW CONFIDENCE / TACTICAL EXECUTION</div>
  <div class="level-row" style="padding:8px 0;">
  <span class="level-label">Market State</span>
  <span class="{lvl['Market Class']}" style="font-size:1rem;letter-spacing:0.5px;">{flow_state}</span>
  </div>
  {flow_rows}
  <div class="level-row" style="padding:7px 0 9px 0;border-top:1px solid #1a3050;margin-top:5px;">
  <span class="level-label">Tactical Execution</span>
  <span class="level-val" style="font-size:0.78rem;color:#c9d1d9;text-align:right;max-width:68%;">{flow_tactical}</span>
  </div>

  <div class="section-divider"></div>
  <div class="section-header">MARKET STATE CLASSIFIER</div>
  <div class="level-row" style="padding:8px 0;">
  <span class="level-label">Market State</span>
  <span class="{lvl['Market Class']}" style="font-size:1rem;letter-spacing:0.5px;">{lvl['Market State']}</span>
  </div>
  <div class="level-row" style="padding:4px 0 8px 0;">
  <span class="level-label">Read</span>
  <span class="level-val" style="font-size:0.78rem;color:#8a9bb0;text-align:right;max-width:65%;">{lvl['Market Read']}</span>
  </div>
  <div class="level-row" style="padding:4px 0 8px 0;">
  <span class="level-label">Tactical Bias</span>
  <span class="level-val" style="font-size:0.78rem;color:#8a9bb0;text-align:right;max-width:65%;">{lvl['Strategy']}</span>
  </div>
 <div class="level-row" style="padding:4px 0 8px 0;">
 <span class="level-label">Data Quality</span>
  <span class="{quality_cls}" style="font-size:0.78rem;text-align:right;max-width:72%;">{escape(lvl.get('Quality Summary', 'n/a'))}</span>
  </div>
 <div class="level-row" style="padding:4px 0 8px 0;">
 <span class="level-label">Chain Quality</span>
 <span class="{chain_quality_cls}" style="font-size:0.78rem;text-align:right;max-width:72%;">{escape(lvl.get('Chain Quality Summary', 'n/a'))}</span>
 </div>
  {warnings_html}
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
  <div class="section-header">LEVEL STRENGTH - HOLD / BREAK RISK</div>
  {strength_html}

  <div class="section-divider"></div>
  <div class="section-header">SECONDARY INTRADAY GEX NODES</div>
  {intraday_html}

  <div class="section-divider"></div>
  <div class="section-header">GEX - REGIME ENGINE</div>
 <div class="level-row"><span class="level-label">Regime</span>
 <span class="{reg_cls}">{lvl['Regime']}</span></div>
 <div class="level-row"><span class="level-label">HVL (Gamma Anchor)</span>
 <span class="level-val lvl-yellow">{lvl['HVL']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">GEX Flip</span>
 <span class="level-val lvl-cyan">{lvl['GEX Flip']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">Call Wall</span>
 <span class="level-val lvl-green">{lvl['Call Wall']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">Secondary Call Node</span>
 <span class="level-val lvl-green">{lvl['Secondary Call Wall']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">Put Wall</span>
 <span class="level-val lvl-red">{lvl['Put Wall']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">Secondary Put Node</span>
 <span class="level-val lvl-red">{lvl['Secondary Put Wall']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">Net GEX</span>
 <span class="level-val">${lvl['Net GEX $B']}B</span></div>

 <div class="section-divider"></div>
  <div class="section-header">DEX - DIRECTION ENGINE</div>
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
  <div class="section-header">VANNA - ACCELERATION / CHARM - TIMING</div>
  <div class="level-row"><span class="level-label">Vanna Regime</span>
  <span class="{van_cls}">{lvl['Vanna Regime']}</span></div>
  <div class="level-row"><span class="level-label">Vanna Fuel</span>
  <span class="level-val lvl-purple">{lvl['Vanna Fuel']}%</span></div>
 <div class="level-row"><span class="level-label">Vanna Flip</span>
 <span class="level-val lvl-purple">{lvl['Vanna Flip']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">Net VEX</span>
 <span class="level-val lvl-purple">${lvl['Net VEX $M']}M</span></div>
 <div class="level-row"><span class="level-label">Charm Flow</span>
 <span class="{chm_cls}">{lvl['Charm Flow']}</span></div>
 <div class="level-row"><span class="level-label">Net CEX / day</span>
 <span class="level-val lvl-orange">{cex_str} delta</span></div>

 <div class="section-divider"></div>
 <div class="section-header">CONTEXT</div>
 <div class="level-row"><span class="level-label">Max Pain</span>
 <span class="level-val lvl-yellow">{lvl['Max Pain']:,.1f}</span></div>
 </div>
 """, unsafe_allow_html=True)

def render_debug_panel(lvl):
    """Collapsible signal state panel shows exactly what the hierarchy sees."""
    d = lvl.get("_debug", {})
    if not d:
        return

    def tick(v):
        return "Yes" if v else "No"

    no_input_warn = "Warning: no live input - Tier 1 and 2 suppressed" if d.get("no_live_input") else "Live inputs active"

    with st.expander("Signal Debug - hierarchy state", expanded=False):
        st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;line-height:2;color:#c9d1d9;
 background:#0d1117;border:1px solid #1e3a1e;border-radius:8px;padding:14px 18px;">
<b style="color:#ffd740">{no_input_warn}</b><br><br>

<b style="color:#4a6070">RAW INPUTS</b><br>
price_change = <b style="color:#{'00e676' if d.get('price_change',0) > 0 else ('ff5252' if d.get('price_change',0) < 0 else '8a9bb0')}">{d.get('price_change', 'n/a')}</b><br>
iv_change = <b style="color:#{'ff5252' if d.get('iv_change',0) > 0 else ('00e676' if d.get('iv_change',0) < 0 else '8a9bb0')}">{d.get('iv_change', 'n/a')}%</b><br><br>

<b style="color:#4a6070">BOOLEAN SIGNALS (strict inequalities)</b><br>
price_up_strict {tick(d.get('price_up_strict'))} (price_change > 0)<br>
iv_falling_strict {tick(d.get('iv_falling_strict'))} (iv_change < 0)<br>
vanna_bullish {tick(d.get('vanna_bullish'))} (from Vanna regime label)<br>
vanna_bearish {tick(d.get('vanna_bearish'))}<br>
vanna_neutral {tick(d.get('vanna_neutral'))}<br><br>

<b style="color:#4a6070">SIGNAL COUNTERS</b><br>
bull_signals = <b style="color:#{'00e676' if d.get('bull_signals',0)==3 else ('ffd740' if d.get('bull_signals',0)==2 else '#c9d1d9')}">{d.get('bull_signals', 0)}</b> / 3 -> Tier {'1 strong' if d.get('bull_signals')==3 else ('2' if d.get('bull_signals')==2 else '3 fallback')}<br>
bear_signals = <b style="color:#{'ff5252' if d.get('bear_signals',0)==3 else ('ffd740' if d.get('bear_signals',0)==2 else '#c9d1d9')}">{d.get('bear_signals', 0)}</b> / 3 -> Tier {'1 strong' if d.get('bear_signals')==3 else ('2' if d.get('bear_signals')==2 else '3 fallback')}<br><br>

<b style="color:#4a6070">GEX/DEX/VEX FLAGS</b><br>
gex_positive {tick(d.get('gex_positive'))} (spot > GEX Flip)<br>
dex_positive {tick(d.get('dex_positive'))} (net DEX > 0)<br>
vex_positive {tick(d.get('vex_positive'))} (net VEX > 0)<br>
cex_positive {tick(d.get('cex_positive'))} (net CEX > 0)<br>
<br>
<b style="color:#4a6070">DATA QUALITY</b><br>
unique_strikes = {d.get('unique_strikes', 'n/a')}<br>
total_oi = {d.get('total_oi', 'n/a')}<br>
strike_span_pct = {d.get('span_pct', 'n/a')}%<br>
near_spot_strikes = {d.get('near_count', 'n/a')}<br>
max_level_cluster = {d.get('max_level_cluster', 'n/a')}<br>
clustered_levels = {d.get('clustered_levels', [])}<br>
oi_proxy = {tick(d.get('oi_proxy'))}<br>
</div>
""", unsafe_allow_html=True)


# UI
st.markdown("""
<div class="gex-header">
  <h1>GEX ENGINE</h1>
  <p>US500 / US100 &nbsp;|&nbsp; GEX / DEX / Vanna / Charm &nbsp;|&nbsp; BackQuant Format</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#0d1117;border:1px solid #1a2a3a;border-radius:8px;padding:10px 16px 6px 16px;
            font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a6070;margin-bottom:12px;">
  CFD SPOT REQUIRED &nbsp;-&nbsp; choose one market, enter one broker CFD spot, then generate levels
</div>
""", unsafe_allow_html=True)

market_choice = st.radio("Market", ["US100", "US500"], horizontal=True)
cfd_spot = st.number_input(f"{market_choice} CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f")

open_price_input = 0.0

btn_col, cc_col = st.columns([4, 1])
with btn_col:
    run = st.button("Generate Levels")
with cc_col:
    if st.button("Clear Cache"):
        fetch_all.clear()
        st.success("Cache cleared")
        st.rerun()

if run:
    if cfd_spot <= 0:
        st.error(f"Enter your {market_choice} CFD spot first.")
        st.stop()

    if market_choice == "US100" and cfd_spot <= 5000:
        st.error("US100 CFD spot looks too low. Enter the full broker price.")
        st.stop()
    if market_choice == "US500" and cfd_spot <= 1000:
        st.error("US500 CFD spot looks too low. Enter the full broker price.")
        st.stop()

    us500_override = cfd_spot if market_choice == "US500" else 0.0
    us100_override = cfd_spot if market_choice == "US100" else 0.0
    selected = [(
        "qqq" if market_choice == "US100" else "spy",
        "us100" if market_choice == "US100" else "us500",
        market_choice,
        open_price_input,
    )]

    with st.spinner("Fetching live spots and computing GEX + Vanna + Charm levels..."):
        try:
            data = fetch_all(us500_override=us500_override, us100_override=us100_override)
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    def recompute_levels(instrument_key, filt_df, spot, open_price, atm_iv, days_val, label):
        open_iv, iv_change, iv_change_pct = store_session_open_iv(instrument_key, atm_iv)
        has_open_price = open_price > 0
        price_change = (spot - open_price) if has_open_price else 0.0
        lvl, _ = levels(filt_df, spot, label, days=days_val,
                        iv_change=iv_change, price_change=price_change,
                        has_open_price=has_open_price)
        return lvl, open_iv, iv_change_pct

    data_key, instrument_key, label, open_price = selected[0]
    item = data[data_key]
    lvl, open_iv, iv_pct = recompute_levels(
        instrument_key, item["filt"], item["index_spot"], open_price,
        item["atm_iv"], dte(item["expiry"]), label
    )
    st.success(f"Levels ready - {datetime.now().strftime('%d %b %Y %H:%M')}")
    st.markdown(f'<span class="source-badge badge-src">{label} converted spot {item["index_spot"]:.1f}</span>', unsafe_allow_html=True)
    iv_dir = "down" if iv_pct < 0 else "up"
    st.markdown(f'ATM IV: **{item["atm_iv"]*100:.1f}%** | Open IV: **{open_iv*100:.1f}%** | IV Change: **{iv_dir} {abs(iv_pct):.1f}%**')
    render_debug_panel(lvl)
    render_level_card(lvl)
    st.markdown("**BackQuant Paste Block:**")
    st.code(backquant_block(label, lvl, item["index_spot"], item["expiry"]), language=None)
    st.pyplot(gex_chart(item["filt"], lvl, item["index_spot"], label))

    st.markdown("""
    <div class="footer">
      Levels cached 30min - Regime engine live (ATM IV / session-open anchor)<br>
      Data is indicative - always verify with your broker before trading
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #3d5268;">
      <div style="font-size: 3rem; margin-bottom: 16px;">GEX</div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;">
        Set spot overrides if needed<br>then click Generate Levels
      </div>
    </div>
    """, unsafe_allow_html=True)
