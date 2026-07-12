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
import time
import random
from scipy.stats import norm
from datetime import datetime, date, timedelta
from html import escape

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

warnings.filterwarnings("ignore")

# YFINANCE RATE-LIMIT RESILIENCE
# Yahoo Finance throttles yfinance's default connection aggressively, and
# Streamlit Community Cloud shares outbound IPs across every app deployed
# on it - so this app can get rate-limited by *other people's* traffic, not
# just its own. Two independent mitigations:
#   1. Impersonate a real browser's TLS/HTTP fingerprint via curl_cffi
#      (yfinance's own maintainers recommend this - plain `requests` gets
#      flagged and blocked far more often). Falls back to a vanilla session
#      with a real User-Agent if curl_cffi isn't installed, so this never
#      hard-crashes the app - but add `curl_cffi` to requirements.txt for
#      the real benefit.
#   2. Retry-with-backoff so a transient 429 self-heals instead of failing
#      the whole "Generate Levels" click.
@st.cache_resource(show_spinner=False)
def _yf_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        import requests
        s = requests.Session()
        s.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36"),
        })
        return s


def session_mode():
    """Which Yahoo-facing session is actually active - curl_cffi's Chrome TLS
 impersonation (hardened) or the plain-requests fallback (spoofed header
 only, much easier for Yahoo to rate-limit). Exists so this is a visible,
 checkable fact in the app instead of a silent, un-debuggable fallback."""
    try:
        import curl_cffi  # noqa: F401
        return "curl_cffi"
    except Exception:
        return "fallback"


def yf_ticker(symbol):
    return yf.Ticker(symbol, session=_yf_session())


def with_retry(fn, tries=5, base_delay=2.0):
    """Retry a Yahoo-touching call with exponential backoff + jitter. Only
    softens transient rate-limit/connection blips - a hard IP-level block
    from Yahoo will still fail after the retries are exhausted, there is
    no code-side fix for that beyond waiting it out."""
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            transient = ("rate" in msg or "429" in msg or "too many" in msg
                        or "timeout" in msg or "connection" in msg)
            if not transient or attempt == tries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, 0.75))
    raise last_exc


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


def session_clock_str():
    """
 "When was this read taken" - tagged with NY time (every session-open
 anchor in this engine, the 0DTE day boundary and the session-open IV
 tracker, is keyed to NY time already) and Uganda local time, plus
 minutes/hours elapsed since the 9:30 ET cash open. This turns "how long
 after the open is this read" into a direct number instead of mental
 timezone math - the opening 30 minutes (9:30-10:00 ET) is materially
 less reliable than later reads because dealer hedging flow hasn't
 settled into a steady rhythm yet, so knowing elapsed-since-open at a
 glance matters.
 """
    ny = market_now("America/New_York")
    local = market_now("Africa/Kampala")  # Uganda - EAT, UTC+3, no DST
    ny_open = ny.replace(hour=9, minute=30, second=0, microsecond=0)
    ny_close = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    if ny.weekday() >= 5:
        session_note = "weekend - NY cash market closed"
    elif ny < ny_open:
        session_note = "pre-market - NY cash session not yet open"
    elif ny > ny_close:
        session_note = "after NY cash close"
    else:
        elapsed_min = int((ny - ny_open).total_seconds() // 60)
        h, m = divmod(elapsed_min, 60)
        session_note = f"{h}h{m:02d}m since NY open" if h else f"{m}m since NY open"
    return (f"{ny.strftime('%d/%m/%Y, %H:%M:%S')} ET | "
           f"{local.strftime('%H:%M:%S')} EAT | {session_note}")


def first_valid_close(ticker_symbol, period="5d"):
    history = with_retry(lambda: yf_ticker(ticker_symbol).history(period=period, auto_adjust=False))
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError(f"No recent close returned for {ticker_symbol}.")
    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if closes.empty:
        raise RuntimeError(f"No valid close returned for {ticker_symbol}.")
    return float(closes.iloc[-1])


def first_expiry(ticker_obj, ticker_symbol):
    """
 First expiry AFTER today, never today itself. If today's date is in the
 expiry list (a 0DTE day - true on SPY/QQQ almost every weekday), returning
 it here would collapse the "aggregate/structural" chain onto the exact
 same single-day chain compute_0dte_levels() already fetches separately,
 so Call Wall/Put Wall/HVL/GEX Flip would just be duplicating the 0DTE
 section instead of showing the broader structural picture.
 """
    expiries = list(with_retry(lambda: getattr(ticker_obj, "options", []) or []))
    if not expiries:
        raise RuntimeError(f"No option expiries returned for {ticker_symbol}.")
    today_str = market_now("America/New_York").strftime("%Y-%m-%d")
    future = [e for e in expiries if e > today_str]
    if not future:
        return expiries[-1]
    return future[0]


def option_chain_frame(ticker_obj, expiry, ticker_symbol):
    chain = with_retry(lambda: ticker_obj.option_chain(expiry))
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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_raw_chain(symbol):
    """
 The only part of the main-chain pipeline that touches Yahoo (etf spot,
 expiry list, option chain). Cached on symbol alone - NOT on the CFD
 spot override - because the override is retyped on essentially every
 intraday click, and keying the cache on it meant every single "Generate
 Levels" click was firing a fresh set of Yahoo requests regardless of
 the 30-minute TTL. The override only rescales strikes locally in
 build_scaled_index() below; it never changes what Yahoo returns.
 """
    ticker = yf_ticker(symbol)
    etf_spot = first_valid_close(symbol)
    expiry = first_expiry(ticker, symbol)
    raw_df = option_chain_frame(ticker, expiry, symbol)
    return {"etf_spot": etf_spot, "expiry": expiry, "raw_df": raw_df}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_raw_ndx_chain():
    """
 NDX cross-check fetch, cached the same way as fetch_raw_chain() - keyed
 on nothing that changes per-click, so it only actually hits Yahoo once
 per 30-minute window regardless of how often spot is updated.

 This is used ONLY for a same-session ATM IV comparison against the
 QQQ-derived read (see ndx_iv_cross_check() below) - it is deliberately
 NOT combined into GEX/DEX/Vanna/Charm math. NDX and QQQ have different
 multipliers and, more importantly, dealers hedge each one with
 different instruments (QQQ shares/baskets vs. whatever NDX market
 makers use) - summing their gamma exposure into one number risks
 double-counting the same economic exposure hedged through correlated
 instruments rather than genuinely adding information. Kept as a
 separate, independent lens instead: something that either agrees with
 the QQQ read (more confidence) or diverges from it (a real flag), never
 blended into a single falsely-precise number.
 """
    ticker = yf_ticker("^NDX")
    ndx_spot = first_valid_close("^NDX")
    expiry = first_expiry(ticker, "^NDX")
    raw_df = option_chain_frame(ticker, expiry, "^NDX")
    return {"ndx_spot": ndx_spot, "expiry": expiry, "raw_df": raw_df}


def ndx_iv_cross_check(qqq_atm_iv):
    """
 Compares QQQ-derived ATM IV against NDX's ATM IV as an independent
 sanity check. NDX is European-style and cash-settled, so it avoids the
 American-exercise/dividend-adjustment distortions that can subtly bias
 QQQ's raw IV - a real, established distinction between ETF options and
 index options, not just a nice-to-have second opinion. Any failure
 here (thin data, rate limit, no listed expiry) degrades to
 "unavailable" rather than raising, so this can never block or break
 the primary QQQ-based read it's checking.
 """
    try:
        raw = fetch_raw_ndx_chain()
        ndx_atm_iv = get_atm_iv(raw["raw_df"], raw["ndx_spot"])
        delta_pp = (ndx_atm_iv - qqq_atm_iv) * 100
        aligned = abs(delta_pp) < 1.5
        return {
            "available": True,
            "ndx_atm_iv": round(ndx_atm_iv * 100, 2),
            "qqq_atm_iv": round(qqq_atm_iv * 100, 2),
            "delta_pp": round(delta_pp, 2),
            "aligned": aligned,
            "note": (
                "ALIGNED - QQQ's IV read looks reliable today"
                if aligned else
                "DIVERGING - QQQ's IV may be distorted (dividend/early-exercise effects); "
                "weight Vanna/Charm signals with extra caution today"
            ),
        }
    except Exception as e:
        return {"available": False, "reason": f"NDX cross-check unavailable: {e}"}


def build_scaled_index(symbol, index_symbol, fallback_scale, override, override_min,
                       display_label):
    has_override = override > override_min
    if not has_override:
        raise RuntimeError(
            f"Enter your live {display_label} CFD spot override so ETF option strikes convert to your broker price."
        )
    raw = fetch_raw_chain(symbol)
    etf_spot = raw["etf_spot"]
    expiry = raw["expiry"]
    raw_df = raw["raw_df"]

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
    ndx_check = (
        ndx_iv_cross_check(atm_iv) if symbol == "QQQ"
        else {"available": False, "reason": "NDX cross-check only applies to US100 (Nasdaq-100) reads."}
    )
    lvl, filt = levels(rich, display_spot, display_label, days=days)
    odte = compute_0dte_levels(symbol, display_spot, scale)
    odte["strength"] = score_0dte_strength(odte, lvl, display_spot)
    block = backquant_block(display_label, lvl, display_spot, expiry, odte=odte, ndx_check=ndx_check)

    return {
        "lvl": lvl,
        "filt": filt,
        "spot": display_spot,
        "index_spot": display_spot,
        "auto_index": display_spot,
        "source_spot": etf_spot,
        "conversion_source": conversion_source,
        "atm_iv": atm_iv,
        "ndx_check": ndx_check,
        "expiry": expiry,
        "scale": round(scale, 4),
        "block": block,
        "odte": odte,
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
def _enrich_core(df, spot, multiplier, T):
    """
 Shared Greeks/exposure engine. Takes T (years) directly so callers can pass
 either the standard days/365 fraction (enrich, below) or a precise sub-day
 0DTE fraction (compute_0dte_levels) without the 1/365 day-floor distorting
 same-day gamma.
 """
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

    # GEX: gamma * OI * multiplier * spot^2 * 0.01 (dollars per 1% underlying
    # move - the standard SpotGamma/SqueezeMetrics convention). Dealer
    # convention applied consistently across every Greek below: dealers are
    # modeled LONG calls (own the call's positive gamma outright, so the
    # call term keeps its natural sign) and SHORT puts (hold the opposite
    # of the put's own greek, so the put term gets negated). This is the
    # same convention that makes "calls add, puts subtract" the textbook
    # Net GEX formula; every exposure column below mirrors it so GEX/DEX/
    # VEX/CEX can never point in dealer-inconsistent directions.
    df["call_gex"] = df["call_gamma"] * df["call_oi"] * multiplier * spot**2 * 0.01
    df["put_gex"] = -df["put_gamma"] * df["put_oi"] * multiplier * spot**2 * 0.01
    df["net_gex"] = df["call_gex"] + df["put_gex"]

    # DEX: delta * OI * multiplier * spot (dollar delta). Long calls keep
    # their own (positive) delta; short puts get the sign flipped from the
    # put's own (negative) delta, which is what makes being short a put a
    # LONG-delta position for the dealer - matching the GEX put-side flip.
    df["call_dex"] = df["call_delta"] * df["call_oi"] * multiplier * spot
    df["put_dex"] = -df["put_delta"] * df["put_oi"] * multiplier * spot
    df["net_dex"] = df["call_dex"] + df["put_dex"]

    # VEX (Vanna Exposure): vanna * OI * multiplier * spot, same long-call/
    # short-put convention as GEX.
    df["call_vex"] = df["call_vanna"] * df["call_oi"] * multiplier * spot
    df["put_vex"] = -df["put_vanna"] * df["put_oi"] * multiplier * spot
    df["net_vex"] = df["call_vex"] + df["put_vex"]

    # CEX (Charm Exposure): per-day delta decay, same long-call/short-put
    # convention as GEX/VEX (previously these two lines had calls and puts
    # swapped relative to that convention - fixed here).
    df["call_cex"] = df["call_charm"] * df["call_oi"] * multiplier * spot / 365
    df["put_cex"] = -df["put_charm"] * df["put_oi"] * multiplier * spot / 365
    df["net_cex"] = df["call_cex"] + df["put_cex"]

    return df


def enrich(df, spot, multiplier, days=30):
    T = max(days / 365.0, 1/365)
    return _enrich_core(df, spot, multiplier, T)


# 0DTE LEVELS
def hours_to_market_close(now_ny=None, close_hour=16):
    """
 Hours remaining today until the 16:00 ET cash close, floored at 5 minutes
 so the gamma formula never divides by a zero/negative T right at or after
 the close (gamma -> infinity as T -> 0 is correct in theory but useless
 in an app - the floor keeps the output stable at the closing bell).
 """
    now_ny = now_ny or market_now("America/New_York")
    close = now_ny.replace(hour=close_hour, minute=0, second=0, microsecond=0)
    remaining_hours = (close - now_ny).total_seconds() / 3600.0
    return max(remaining_hours, 5.0 / 60.0)


def odte_time_fraction(now_ny=None):
    """
 0DTE time-to-expiry in years, on the same calendar-day annualised basis
 (T = days/365) used everywhere else in this engine, so 0DTE gamma sits on
 the same scale as the standard-expiry gamma rather than a different
 (trading-hour) convention that would make the two non-comparable.
 """
    return hours_to_market_close(now_ny) / (24.0 * 365.0)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_raw_0dte_chain(ticker_symbol, today_str):
    """
 The only part of the 0DTE pipeline that touches Yahoo. Cached per symbol
 per calendar day - independent of the live CFD spot override, so
 retyping spot (which changes on every intraday click) never forces a
 refetch of a chain that hasn't actually changed. See fetch_raw_chain()
 for the same pattern on the main chain.
 """
    try:
        ticker = yf_ticker(ticker_symbol)
        available_expiries = list(with_retry(lambda: getattr(ticker, "options", []) or []))
        if today_str not in available_expiries:
            return {
                "available": False,
                "reason": f"No same-day (0DTE) expiry listed for {ticker_symbol} today.",
            }
        raw_df = option_chain_frame(ticker, today_str, ticker_symbol)
        return {"available": True, "raw_df": raw_df}
    except Exception as e:
        return {"available": False, "reason": f"0DTE fetch failed for {ticker_symbol}: {e}"}


def compute_0dte_levels(ticker_symbol, display_spot, scale):
    """
 Fetch the same-day expiry chain (if the underlying lists one today), scale
 its strikes with the same live ETF->CFD ratio used for the main chain, and
 compute the 0DTE Call Wall / 0DTE Put Wall the way BackQuant's "0DTE
 LEVELS" group does it: strike of maximum call gamma exposure at/above
 spot, and strike of maximum (most negative) put gamma exposure at/below
 spot. Same wall convention as the standard Call Wall / Put Wall above -
 just restricted to today's expiry and today's precise time-to-close
 instead of the main chain's DTE. The Yahoo fetch itself lives in
 fetch_raw_0dte_chain() above; everything here is local math on that
 cached result, safe to rerun on every spot update at zero network cost.
 """
    now_ny = market_now("America/New_York")
    today_str = now_ny.strftime("%Y-%m-%d")
    raw = fetch_raw_0dte_chain(ticker_symbol, today_str)
    if not raw.get("available"):
        return raw
    try:
        scaled_0dte = raw["raw_df"].copy()
        scaled_0dte["strike"] = (scaled_0dte["strike"] * scale).round(1)

        T_0dte = odte_time_fraction(now_ny)
        rich_0dte = _enrich_core(scaled_0dte, display_spot, EQ_MULTIPLIER, T_0dte)
        atm_iv_0dte = get_atm_iv(scaled_0dte, display_spot)

        by_strike = rich_0dte.groupby("strike", sort=True).sum(numeric_only=True)
        if by_strike.empty:
            return {
                "available": False,
                "reason": f"0DTE chain for {ticker_symbol} returned no usable strikes.",
            }

        call_gex = by_strike["call_gex"]
        put_gex = by_strike["put_gex"]
        call_above = call_gex[call_gex.index >= display_spot]
        put_below = put_gex[put_gex.index <= display_spot]
        call_wall = float(call_above.idxmax()) if not call_above.empty else float(call_gex.idxmax())
        put_wall = float(put_below.idxmin()) if not put_below.empty else float(put_gex.idxmin())

        # Diagnostics feeding score_0dte_strength(): how much of the day's
        # 0DTE gamma actually sits at the wall strike (vs. it just edging
        # out neighbours on the coarse post-scale grid), and how much real
        # OI backs it (thin/zero 0DTE OI early in the session is a data
        # quality issue, not a genuine wall).
        call_gex_abs_total = float(call_gex.abs().sum())
        put_gex_abs_total = float(put_gex.abs().sum())
        call_gex_at_wall = float(abs(call_gex.loc[call_wall])) if call_wall in call_gex.index else 0.0
        put_gex_at_wall = float(abs(put_gex.loc[put_wall])) if put_wall in put_gex.index else 0.0
        call_dominance = (call_gex_at_wall / call_gex_abs_total) if call_gex_abs_total > 0 else 0.0
        put_dominance = (put_gex_at_wall / put_gex_abs_total) if put_gex_abs_total > 0 else 0.0
        call_oi_at_wall = float(by_strike.loc[call_wall, "call_oi"]) if call_wall in by_strike.index else 0.0
        put_oi_at_wall = float(by_strike.loc[put_wall, "put_oi"]) if put_wall in by_strike.index else 0.0
        net_gex_0dte = float(by_strike["net_gex"].sum())

        return {
            "available": True,
            "expiry": today_str,
            "call_wall": round(call_wall, 1),
            "put_wall": round(put_wall, 1),
            "atm_iv": round(atm_iv_0dte, 4),
            "hours_to_close": round(T_0dte * 24 * 365, 2),
            "call_dominance": round(call_dominance, 4),
            "put_dominance": round(put_dominance, 4),
            "call_oi_at_wall": round(call_oi_at_wall, 1),
            "put_oi_at_wall": round(put_oi_at_wall, 1),
            "net_gex_0dte_m": round(net_gex_0dte / 1e6, 2),
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"0DTE compute failed for {ticker_symbol}: {e}",
        }


def score_0dte_strength(odte, aggregate_lvl, spot):
    """
 HOLD/BREAK confidence for the 0DTE Call Wall and 0DTE Put Wall, on the same
 0-100 scale and HOLD LIKELY / TEST-RESPECT / BREAK RISK verdicts as
 score_level_strength() below, built from signals specific to a same-day
 chain rather than reusing the aggregate scorer's exposure columns directly:

 - Dominance (0-45): this strike's share of total same-side 0DTE gamma in
   the chain. A wall backed by most of the day's gamma is structurally
   real; one that barely edges out its neighbours on the coarse post-scale
   strike grid (see the grid-spacing caveat from the 0DTE build) is not.
   This is the primary driver - it's the only signal that's actually
   specific to *this* expiry's same-day positioning.
 - Liquidity (0-25): OI actually sitting at the wall strike. Thin/zero
   0DTE OI early in the session is a real data-quality issue, not a wall.
 - Distance to spot (0-15): closer = more immediately live/relevant today.
 - Aggregate confluence (0-15): how close the 0DTE wall sits to the
   structural engine's Call Wall / Put Wall / GEX Flip / HVL. Real signal,
   but a MINOR one - two unrelated expiries' OI structures merely sitting
   in the same neighbourhood isn't the same thing as same-day conviction,
   so this is capped low and decays fast (full credit only within ~0.3%
   of spot) rather than being able to carry a weak-dominance strike into
   HOLD LIKELY on proximity alone.
 """
    if not odte or not odte.get("available"):
        return []

    aggregate_targets = {
        "Call Wall": aggregate_lvl.get("Call Wall"),
        "Put Wall": aggregate_lvl.get("Put Wall"),
        "GEX Flip": aggregate_lvl.get("GEX Flip"),
        "HVL": aggregate_lvl.get("HVL"),
    }

    def nearest_aggregate(level):
        best_name, best_dist = None, None
        for name, target in aggregate_targets.items():
            if target is None:
                continue
            d = abs(float(level) - float(target))
            if best_dist is None or d < best_dist:
                best_dist, best_name = d, name
        return best_name, best_dist

    scored = []
    sides = [
        ("0DTE Call", odte["call_wall"], odte.get("call_dominance", 0.0), odte.get("call_oi_at_wall", 0.0)),
        ("0DTE Put", odte["put_wall"], odte.get("put_dominance", 0.0), odte.get("put_oi_at_wall", 0.0)),
    ]
    for name, level, dominance, oi_at_wall in sides:
        dominance_score = min(45.0, dominance * 120.0)
        liquidity_score = min(25.0, (oi_at_wall / 50.0) * 25.0)
        distance_pct = abs(level - spot) / max(spot, 1.0)
        distance_score = max(0.0, 15.0 - min(15.0, distance_pct * 300.0))
        agg_name, agg_dist = nearest_aggregate(level)
        agg_dist_pct = (agg_dist / max(spot, 1.0)) if agg_dist is not None else 1.0
        confluence_score = max(0.0, 15.0 - min(15.0, agg_dist_pct * 1500.0))

        score = int(round(min(100.0, dominance_score + liquidity_score + distance_score + confluence_score)))
        if score >= 70:
            verdict, cls = "HOLD LIKELY", "regime-blue"
        elif score >= 45:
            verdict, cls = "TEST / RESPECT", "regime-orange"
        else:
            verdict, cls = "BREAK RISK", "regime-neg"

        notes = []
        if agg_name and agg_dist is not None:
            notes.append(f"aligns w/ aggregate {agg_name} (delta {agg_dist:,.1f})")
        else:
            notes.append("no aggregate confluence nearby")
        if oi_at_wall < 50:
            notes.append("thin 0DTE OI - early session, weight down")

        scored.append({
            "name": name,
            "level": round(level, 1),
            "score": score,
            "verdict": verdict,
            "class": cls,
            "note": " | ".join(notes),
            "dominance_pct": round(dominance * 100, 1),
        })
    return scored


def _zero_cross_level(strikes, values, spot=None):
    """
 Interpolated zero-crossing strike of a cumulative exposure curve. When the
 curve crosses zero more than once (common on a noisy real chain), picks
 the crossing whose interpolated level sits NEAREST TO SPOT rather than
 the crossing in the middle of the list - the flip that matters to a
 trader is the one closest to where price actually is right now, not
 whichever crossing happens to fall at the midpoint index of however many
 sign changes the chain produced.
 """
    strikes = np.asarray(strikes, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(strikes) == 0:
        return 0.0
    crossings = np.where(np.diff(np.sign(values)))[0]
    if len(crossings) > 0:
        candidates = []
        for i in crossings:
            j = min(i + 1, len(strikes) - 1)
            v0, v1 = values[i], values[j]
            if v1 != v0:
                candidates.append(strikes[i] - v0 * (strikes[j] - strikes[i]) / (v1 - v0))
            else:
                candidates.append(strikes[i])
        candidates = np.asarray(candidates, dtype=float)
        if spot is not None:
            return float(candidates[np.argmin(np.abs(candidates - spot))])
        return float(candidates[len(candidates) // 2])
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


def classify_regime(gex_positive, dex_positive, vanna_bullish, vanna_bearish, vanna_fuel,
                    cex_positive, days, exposure_stats, data_quality,
                    has_open_price=False, price_confirms=None, iv_confirms=None):
    """
 ONE authoritative intraday-bias verdict. This replaces what used to be
 two separately-computed classifiers (a binary GEX/DEX/Vanna decision tree
 and a weighted-score classifier) plus a third live-price/IV vote-count
 system driving a fourth "Entry Trigger" - four paths that read mostly the
 same inputs through different branching logic and could genuinely
 disagree with each other on screen at the same time. GEX sets the Primary
 Regime, DEX sets the Directional Bias, Vanna/Charm are confidence inputs
 and nuance only - never a competing verdict. Live price/IV momentum, when
 a real session-open price is entered, nudges confidence rather than
 producing its own separate label.
 """
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
    raw_score = gex_score * 0.40 + dex_score * 0.35 + vanna_score * 0.15 + charm_score * 0.10

    # Live price/IV momentum nudges confidence instead of casting its own
    # vote - this is what used to drive a separate Intraday Condition/Entry
    # Trigger vote count that could contradict the GEX/DEX-led verdict.
    momentum_note = None
    if has_open_price and price_confirms is not None:
        agree = price_confirms == dex_positive
        raw_score += 8 if agree else -10
        momentum_note = ("Live price action confirms DEX bias." if agree
                        else "Live price action is fighting DEX bias - wait for confirmation.")
        if iv_confirms is not None:
            iv_agree = iv_confirms == dex_positive
            raw_score += 4 if iv_agree else -6

    # Chain quality ACTIVELY discounts confidence - previously this only
    # appended a text warning and never moved the number itself.
    quality_multiplier = {"OK": 1.00, "WATCH": 0.85, "LOW": 0.65}.get(data_quality, 0.85)
    overall_score = clamp(raw_score * quality_multiplier)
    flow_quality = "OK" if overall_score >= 70 else ("WATCH" if overall_score >= 50 else "LOW")

    primary_regime = "POSITIVE GAMMA" if gex_positive else "NEGATIVE GAMMA"
    directional_bias = "BULLISH" if dex_positive else "BEARISH"

    if gex_positive and dex_positive:
        state = "CONTROLLED BULLISH"
        tactic = "Buy pullbacks into support; take profits into call walls; avoid chasing extensions while gamma is pinning."
    elif gex_positive and not dex_positive:
        state = "PINNED RANGE"
        tactic = "Fade extremes first; sell failed rallies near resistance; avoid aggressive breakout trades; expect pinning near HVL / Max Pain."
    elif (not gex_positive) and dex_positive:
        if vanna_bullish and vanna_score >= 60:
            state = "SQUEEZE RISK"
            tactic = "Dealers may chase upside on a break; favor momentum longs after reclaimed resistance; avoid fading a clean breakout."
        else:
            state = "BULLISH EXPANSION WATCH"
            tactic = "Favor momentum only after reclaimed resistance; use DEX confirmation; avoid fading a clean breakout."
    else:
        if vanna_bearish and vanna_score >= 60:
            state = "CRASH RISK"
            tactic = "Respect downside breaks; rallies likely fail; avoid mean-reversion longs until price reclaims key gamma levels."
        else:
            state = "BEARISH EXPANSION"
            tactic = "Sell failed rallies; respect downside breaks; avoid mean-reversion longs until price reclaims key gamma levels."

    if momentum_note:
        tactic = f"{tactic} {momentum_note}"
    if data_quality != "OK":
        tactic = f"{tactic} Chain quality is {data_quality} (confidence already discounted for this) - reduce size until levels confirm."

    state_cls = "regime-pos" if directional_bias == "BULLISH" else "regime-neg"
    if state == "BULLISH EXPANSION WATCH" and overall_score < 60:
        state_cls = "regime-orange"

    # Alternative Scenario: flip whichever of GEX/DEX contributed the LEAST
    # confidence (the least decisive signal), tied to a real probability
    # rather than a decorative afterthought.
    if gex_score <= dex_score:
        alt_gex_positive, alt_dex_positive = (not gex_positive), dex_positive
    else:
        alt_gex_positive, alt_dex_positive = gex_positive, (not dex_positive)
    alt_state = {
        (True, True): "CONTROLLED BULLISH",
        (True, False): "PINNED RANGE",
        (False, True): "SQUEEZE RISK" if (vanna_bullish and vanna_score >= 60) else "BULLISH EXPANSION WATCH",
        (False, False): "CRASH RISK" if (vanna_bearish and vanna_score >= 60) else "BEARISH EXPANSION",
    }[(alt_gex_positive, alt_dex_positive)]
    alt_probability = clamp(100 - overall_score, lo=5, hi=45)

    is_bullish = directional_bias == "BULLISH"
    if overall_score >= 70:
        entry_trigger = f"{'LONG' if is_bullish else 'SHORT'} CONFIRMED - {state} at {overall_score}% confidence"
        trigger_cls = "regime-pos" if is_bullish else "regime-neg"
    elif overall_score >= 50:
        entry_trigger = f"{'LONG' if is_bullish else 'SHORT'} LEAN - confirm with price action ({overall_score}% confidence)"
        trigger_cls = "regime-orange"
    else:
        entry_trigger = f"WAIT - conflicted / low confidence ({overall_score}%) - alt scenario {alt_state} at {alt_probability}%"
        trigger_cls = "regime-purple"

    gex_note_1 = "Mean reversion dominant" if gex_positive else "Expansion risk dominant"
    gex_note_2 = "Breakout probability suppressed" if gex_positive else "Breakouts can extend faster"
    dex_note_1 = "Upside pressure exists" if dex_positive else "Downside pressure exists"
    dex_note_2 = "Expansion capped by gamma" if gex_positive else "Expansion enabled by gamma"
    if vanna_bullish:
        vanna_name, vanna_note_1 = "VANNA FUEL", "Squeeze support active"
        vanna_note_2 = "Vol crush favorable" if gex_positive else "Can accelerate with trend"
    elif vanna_bearish:
        vanna_name, vanna_note_1 = "VANNA DRAG", "Downside acceleration risk"
        vanna_note_2 = "Respect failed supports" if not gex_positive else "Gamma may slow expansion"
    else:
        vanna_name, vanna_note_1 = "VANNA NEUTRAL", "Acceleration fuel limited"
        vanna_note_2 = "Do not force momentum reads"
    charm_note_1 = "Small bullish close bias" if cex_positive else "Small bearish close bias"
    charm_note_2 = "Low weight due to DTE" if days >= 21 else "Useful for timing only"

    hierarchy = [
        {"name": primary_regime, "score": gex_score, "note_1": gex_note_1, "note_2": gex_note_2},
        {"name": "BULLISH DELTA" if dex_positive else "BEARISH DELTA", "score": dex_score, "note_1": dex_note_1, "note_2": dex_note_2},
        {"name": vanna_name, "score": vanna_score, "note_1": vanna_note_1, "note_2": vanna_note_2},
        {"name": "CHARM DRIFT", "score": charm_score, "note_1": charm_note_1, "note_2": charm_note_2},
    ]

    return {
        "primary_regime": primary_regime,
        "directional_bias": directional_bias,
        "state": state,
        "state_cls": state_cls,
        "tactical": tactic,
        "confidence": overall_score,
        "quality": flow_quality,
        "summary": f"{flow_quality} - {overall_score}% confidence ({primary_regime} / {directional_bias})",
        "alt_scenario": alt_state,
        "alt_probability": alt_probability,
        "entry_trigger": entry_trigger,
        "trigger_cls": trigger_cls,
        "hierarchy": hierarchy,
    }


def score_level_strength(by_strike, spot, level_map, gex_positive, dex_positive, pain_curve=None):
    """
 Estimate whether each major level is likely to hold or break, using a
 genuine per-level DOMINANCE as the primary driver - the same "how much
 of the same-side exposure actually sits here" concept that turned out to
 matter most for the 0DTE walls (score_0dte_strength), now applied
 consistently to every level instead of being 0DTE-only. Different level
 types get dominance measured against whatever actually defines them,
 rather than force-fitting one formula onto concepts that aren't all the
 same kind of thing:

 - Call/Put Wall, Call/Put DEX Wall: same-side (call or put, above or
   below spot) share of gamma/delta exposure concentrated at that exact
   strike - directly mirrors the 0DTE wall dominance calculation.
 - HVL: share of TOTAL chain |net_gex| sitting at that single strike (HVL
   is defined as the single largest-magnitude gamma strike, so its natural
   dominance is against the whole chain, not one side).
 - GEX Flip: has no OI concentrated at it by definition - it's a
   zero-crossing, not a wall - so its dominance analogue is FLIP
   CONVICTION: how steep the cumulative-GEX curve is right at the
   crossing. A flip that swings from firmly negative to firmly positive
   over a couple of strikes is a real regime boundary; one that drifts
   gradually across dozens of strikes is a coin flip price can wander
   through either way.
 - Max Pain: also has no concentration by definition (it can sit at a
   strike with modest OI), so its analogue is PAIN CONVICTION: how much
   more total payout the next-best candidate strike would cost option
   writers. A sharp, unique minimum is a real magnet; a flat minimum
   (several strikes within a hair of the same total payout) barely pulls
   price anywhere.

 Confluence and regime-alignment are still in the mix but capped low
 (15 pts each) and confluence decays fast - full credit only within
 ~0.3% of spot - so two levels merely sitting in the same neighbourhood
 can no longer carry a weak-dominance level into HOLD LIKELY on
 proximity alone, the same overweighting problem fixed above in
 score_0dte_strength.
 """
    strikes = by_strike.index.to_numpy(dtype=float)

    def nearest_idx(level):
        return int(np.argmin(np.abs(strikes - float(level))))

    def side_dominance(col, side, idx):
        if col not in by_strike:
            return 0.0
        series_abs = by_strike[col].abs().to_numpy(dtype=float)
        if side == "above":
            mask = strikes >= spot
        elif side == "below":
            mask = strikes <= spot
        else:
            mask = np.ones_like(strikes, dtype=bool)
        total = series_abs[mask].sum()
        return float(series_abs[idx] / total) if total > 0 else 0.0

    cum_gex = by_strike["net_gex"].cumsum().to_numpy(dtype=float) if "net_gex" in by_strike else np.zeros_like(strikes)
    gex_span = float(np.ptp(cum_gex)) if len(cum_gex) > 1 else 0.0

    def flip_conviction(level):
        idx = nearest_idx(level)
        lo, hi = max(0, idx - 1), min(len(strikes) - 1, idx + 1)
        if hi == lo or gex_span <= 0:
            return 0.0
        return float(min(1.0, abs(cum_gex[hi] - cum_gex[lo]) / gex_span))

    pain_conviction_val = 0.0
    if pain_curve is not None:
        _, pa_losses = pain_curve
        pa_losses = np.asarray(pa_losses, dtype=float)
        if len(pa_losses) > 1:
            order = np.argsort(pa_losses)
            best, second = pa_losses[order[0]], pa_losses[order[1]]
            curve_range = float(pa_losses.max() - pa_losses.min())
            pain_conviction_val = float(min(1.0, (second - best) / curve_range)) if curve_range > 0 else 0.0

    level_family = {
        "HVL": "hvl", "GEX Flip": "flip", "Call Wall": "call_gex_wall", "Put Wall": "put_gex_wall",
        "Call DEX Wall": "call_dex_wall", "Put DEX Wall": "put_dex_wall", "Max Pain": "pain",
    }
    levels_f = {k: float(v) for k, v in level_map.items()}

    scored = []
    for name, level in levels_f.items():
        idx = nearest_idx(level)
        family = level_family.get(name, "other")
        oi_at_level = 0.0

        if name == "Call Wall":
            dominance, dom_label = side_dominance("call_gex", "above", idx), "call-side GEX"
            oi_at_level = float(by_strike["call_oi"].iloc[idx]) if "call_oi" in by_strike else 0.0
        elif name == "Put Wall":
            dominance, dom_label = side_dominance("put_gex", "below", idx), "put-side GEX"
            oi_at_level = float(by_strike["put_oi"].iloc[idx]) if "put_oi" in by_strike else 0.0
        elif name == "Call DEX Wall":
            dominance, dom_label = side_dominance("call_dex", "above", idx), "call-side DEX"
            oi_at_level = float(by_strike["call_oi"].iloc[idx]) if "call_oi" in by_strike else 0.0
        elif name == "Put DEX Wall":
            dominance, dom_label = side_dominance("put_dex", "below", idx), "put-side DEX"
            oi_at_level = float(by_strike["put_oi"].iloc[idx]) if "put_oi" in by_strike else 0.0
        elif name == "HVL":
            dominance, dom_label = side_dominance("net_gex", "all", idx), "total chain GEX"
            oi_at_level = float(by_strike["call_oi"].iloc[idx] + by_strike["put_oi"].iloc[idx]) if "call_oi" in by_strike else 0.0
        elif name == "GEX Flip":
            dominance, dom_label = flip_conviction(level), "flip steepness"
        elif name == "Max Pain":
            dominance, dom_label = pain_conviction_val, "payout-curve sharpness"
        else:
            dominance, dom_label = 0.0, "n/a"

        dominance_score = min(45.0, dominance * 100.0)
        liquidity_score = min(15.0, (oi_at_level / 50.0) * 15.0) if family not in ("flip", "pain") else 7.5
        distance_pct = abs(level - spot) / max(spot, 1.0)
        distance_score = max(0.0, 15.0 - min(15.0, distance_pct * 300.0))
        cluster = sum(
            1 for other_name, other_val in levels_f.items()
            if other_name != name and abs(other_val - level) <= max(spot * 0.003, 5.0)
        )
        confluence_score = min(15.0, cluster * 7.5)

        is_resistance = "Call" in name or level > spot
        is_support = "Put" in name or level < spot
        flow_score = 0.0
        if gex_positive:
            flow_score += 5.0
        if is_support and dex_positive:
            flow_score += 5.0
        if is_resistance and not dex_positive:
            flow_score += 5.0

        score = int(round(min(100.0, dominance_score + liquidity_score + distance_score + confluence_score + flow_score)))
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
            "dominance_pct": round(dominance * 100, 1),
            "dominance_label": dom_label,
        })
    return scored


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
    real_oi_mask = (df["call_oi"] > 0) | (df["put_oi"] > 0)
    if real_oi_mask.any():
        df = df[real_oi_mask].copy()
    else:
        oi_proxy = True
        df = df.copy()
        df["call_oi"] = np.where(df["call_iv"] > 0.01, 1.0, 0.0)
        df["put_oi"] = np.where(df["put_iv"] > 0.01, 1.0, 0.0)
        if not ((df["call_oi"] > 0) | (df["put_oi"] > 0)).any():
            df["call_oi"] = 1.0
            df["put_oi"] = 1.0
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
    gflip = _zero_cross_level(strikes, by_strike["net_gex"].cumsum().to_numpy(), spot=spot)
    gflip = float(np.clip(gflip, min(put_wall, call_wall), max(put_wall, call_wall)))

    call_dex = by_strike["call_dex"]
    put_dex = by_strike["put_dex"]
    call_dex_above = call_dex[call_dex.index >= spot]
    put_dex_below = put_dex[put_dex.index <= spot]
    call_delta_wall = float(call_dex_above.idxmax()) if not call_dex_above.empty else float(call_dex.idxmax())
    put_delta_wall = float(put_dex_below.idxmin()) if not put_dex_below.empty else float(put_dex.idxmin())
    dflip = _zero_cross_level(strikes, by_strike["net_dex"].cumsum().to_numpy(), spot=spot)
    dflip = float(np.clip(dflip, min(put_delta_wall, call_delta_wall), max(put_delta_wall, call_delta_wall)))

    vanna_flip = _zero_cross_level(strikes, by_strike["net_vex"].cumsum().to_numpy(), spot=spot)

    sa = filt["strike"].to_numpy(dtype=float)
    coi = filt["call_oi"].to_numpy(dtype=float)
    poi = filt["put_oi"].to_numpy(dtype=float)
    # Max Pain = settlement price minimizing total intrinsic-value payout to
    # option holders. Call payout is max(settle-strike,0) weighted by call
    # OI; put payout is max(strike-settle,0) weighted by put OI (previously
    # swapped, which pulled Max Pain toward the wrong side of the chain).
    losses = [np.sum(np.maximum(s - sa, 0) * coi) + np.sum(np.maximum(sa - s, 0) * poi) for s in sa]
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
    if days >= 21:
        charm_note = f"CHARM: {'up' if cex_positive else 'down'} (low weight - {days}+ DTE)"
    else:
        # Charm's "afternoon" framing only means something if it's actually
        # checked against the clock - previously this said "AFTERNOON" no
        # matter what time it was (even at midnight on a weekend). Charm
        # decay is a real but slow, structural drift that's most pronounced
        # late in the session, not a same-session retracement predictor -
        # word it as a forward expectation before the fact and a live
        # condition only once it's actually true.
        ny_hour = market_now("America/New_York").hour
        direction = "melt-up" if cex_positive else "fade"
        if ny_hour >= 14:
            charm_note = f"AFTERNOON {direction.upper()} BIAS - charm pressure is live now"
        else:
            charm_note = f"{direction.upper()} BIAS (charm) - typically most pronounced late in the session, not yet"

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

    # Live price/IV momentum, when a real session-open price was entered,
    # feeds classify_regime() as a confidence nudge - it no longer casts a
    # separate vote that could contradict the GEX/DEX-led verdict.
    price_confirms = (price_change > 0) if has_open_price else None
    iv_confirms = (iv_change < 0) if has_open_price else None

    def exposure_balance(col, total):
        gross = float(filt[col].abs().sum()) if col in filt else 0.0
        return 0.0 if gross <= 0 else min(1.0, abs(float(total)) / gross)

    verdict = classify_regime(
        gex_positive, dex_positive, vanna_bullish, vanna_bearish, vanna_fuel,
        cex_positive, days,
        {
            "gex_balance": exposure_balance("net_gex", total_net_gex),
            "dex_balance": exposure_balance("net_dex", total_net_dex),
            "cex_balance": exposure_balance("net_cex", total_net_cex),
            "gex_distance_pct": abs(spot - gflip) / max(spot, 1.0) * 100.0,
        },
        quality["confidence"],
        has_open_price=has_open_price,
        price_confirms=price_confirms,
        iv_confirms=iv_confirms,
    )
    level_strength = score_level_strength(by_strike, spot, {
        "HVL": hvl,
        "GEX Flip": gflip,
        "Call Wall": call_wall,
        "Put Wall": put_wall,
        "Call DEX Wall": call_delta_wall,
        "Put DEX Wall": put_delta_wall,
        "Max Pain": max_pain,
    }, gex_positive, dex_positive, pain_curve=(sa, losses))

    em_lo, em_hi = expected_move(spot, total_net_gex / 1e9, days)

    return {
        "HVL": round(hvl, 1),
        "Call Wall": round(call_wall, 1),
        "Put Wall": round(put_wall, 1),
        "GEX Flip": round(gflip, 1),
        "Call DEX Wall": round(call_delta_wall, 1),
        "Put DEX Wall": round(put_delta_wall, 1),
        "DEX Flip": round(dflip, 1),
        "Vanna Flip": round(vanna_flip, 1),
        "Max Pain": round(max_pain, 1),
        "Expected Move Lo": em_lo,
        "Expected Move Hi": em_hi,
        "DTE": days,
        "Net GEX $B": round(total_net_gex / 1e9, 3),
        "Net DEX $B": round(total_net_dex / 1e9, 3),
        "Net VEX $M": round(total_net_vex / 1e6, 2),
        "Net CEX (daily)": round(total_net_cex, 0),
        "Regime": regime,
        "DEX Bias": dex_bias,
        "Vanna Regime": vanna_regime,
        "Charm Flow": charm_flow,
        "Vanna Fuel": vanna_fuel,
        "Level Strength": level_strength,
        "Flow Confidence": verdict,
        "Primary Regime": verdict["primary_regime"],
        "Directional Bias": verdict["directional_bias"],
        "Market State": verdict["state"],
        "Market Class": verdict["state_cls"],
        "Tactical Execution": verdict["tactical"],
        "Confidence": verdict["confidence"],
        "Alt Scenario": verdict["alt_scenario"],
        "Alt Probability": verdict["alt_probability"],
        "Vanna Note": vanna_note,
        "Charm Note": charm_note,
        "Entry Trigger": verdict["entry_trigger"],
        "Trigger Class": verdict["trigger_cls"],
        "IV Change": round(iv_change * 100, 2),
        "Price Change": round(price_change, 2),
        "Data Quality": verdict["quality"],
        "Quality Summary": verdict["summary"],
        "Chain Quality": quality["confidence"],
        "Chain Quality Summary": quality["summary"],
        "Quality Warnings": quality["warnings"],
        "_debug": {
            "price_change": round(price_change, 4),
            "iv_change": round(iv_change * 100, 4),
            "price_confirms": price_confirms,
            "iv_confirms": iv_confirms,
            "has_open_price": has_open_price,
            "vanna_bullish": vanna_bullish,
            "vanna_bearish": vanna_bearish,
            "vanna_neutral": vanna_neutral,
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

def build_quick_read(lvl, odte, spot):
    """
 Four things, in this order, nothing else: the single level most likely
 to get reacted on today (0DTE preferred - freshest, most immediate
 conviction - falling back to the best aggregate level on days without a
 same-day expiry), the magnet (Max Pain/HVL gravity, a separate
 mechanic from wall dominance), market state, and an honest
 retracement-risk read that explicitly says when a clean expansion
 without a retest first is plausible, not just when chop is likely.
 This is deliberately narrower than earlier versions - the full
 Resistance/Support/dominance breakdown already lives in Level
 Strength below; Quick Read's job is the fast read, not a second copy
 of the detailed one.
 """
    pool_0dte = []
    if odte and odte.get("available"):
        for item in odte.get("strength", []):
            pool_0dte.append({"name": item["name"], "level": item["level"], "dominance_pct": item["dominance_pct"]})
    pool_agg = [
        {"name": item["name"], "level": item["level"], "dominance_pct": item["dominance_pct"]}
        for item in lvl.get("Level Strength", [])
    ]
    if pool_0dte:
        key_level, key_source = max(pool_0dte, key=lambda x: x["dominance_pct"]), "0DTE"
    elif pool_agg:
        key_level, key_source = max(pool_agg, key=lambda x: x["dominance_pct"]), "structural"
    else:
        key_level, key_source = None, None

    gex_positive = lvl.get("_debug", {}).get("gex_positive", True)
    bias = lvl.get("Directional Bias", "BULLISH")
    vanna_fuel = lvl.get("Vanna Fuel", 0)

    # --- Magnet: HVL and Max Pain are gravity-center concepts, distinct
    # from wall/dominance mechanics (which are about resistance/support at
    # OI concentration points). This is the answer to "why does a
    # low-dominance level sometimes react harder than a high-dominance
    # one" - it isn't the wall pulling price, it's this, a completely
    # separate mechanic. Max Pain is the classic "gravitational pull
    # toward a settlement price" concept; when HVL (the single largest
    # gamma concentration) coincides with it, that pull is reinforced
    # rather than being two separate, weaker claims. Strength depends on
    # regime AND time-to-expiry: positive gamma pinning is a real,
    # actively-defended pull; negative gamma has no such defending force,
    # so the pull there is only as real as classic pin-risk dynamics
    # close to expiry - genuinely weak further out.
    hvl_level = lvl.get("HVL")
    max_pain_level = lvl.get("Max Pain")
    dte = lvl.get("DTE", 30)
    magnet_name, magnet_level, magnet_direction = None, None, None
    magnet_strength, magnet_opposes = None, False
    if max_pain_level is not None:
        magnet_level = max_pain_level
        magnet_name = "Max Pain"
        tol = max(spot * 0.0015, 3.0)
        if hvl_level is not None and abs(hvl_level - max_pain_level) <= tol:
            magnet_name = "HVL / Max Pain"
        magnet_direction = "above" if magnet_level > spot else ("below" if magnet_level < spot else "at spot")
        if gex_positive:
            magnet_strength = "STRONG"
        elif dte <= 2:
            magnet_strength = "MODERATE"
        else:
            magnet_strength = "WEAK"
        if (magnet_direction == "below" and bias == "BULLISH") or (magnet_direction == "above" and bias == "BEARISH"):
            magnet_opposes = True

    # --- Retracement risk: negative gamma is the real structural driver of
    # chop before a directional move confirms, because it lacks the
    # defending hedging force positive gamma has. A magnet pulling
    # against the current bias is a second, independent reason chop can
    # show up before the move confirms. When NEITHER factor is present,
    # say so plainly - a clean expansion without testing lower levels
    # first is a real, distinct possibility worth naming, not just the
    # absence of a warning.
    if gex_positive:
        retrace_tier = "NORMAL"
        retrace_note = "positive gamma - a pullback into support is the expected base case here, not a warning sign"
    else:
        ny = market_now("America/New_York")
        ny_open = ny.replace(hour=9, minute=30, second=0, microsecond=0)
        minutes_since_open = (ny - ny_open).total_seconds() / 60.0 if ny >= ny_open else None
        early_session = minutes_since_open is not None and minutes_since_open < 60
        high_vanna = vanna_fuel >= 60
        magnet_pull = magnet_opposes and magnet_strength in ("STRONG", "MODERATE")

        reasons = []
        if early_session:
            reasons.append("early session")
        if high_vanna:
            reasons.append("elevated vanna fuel")
        if magnet_pull:
            reasons.append(f"{magnet_name} pulling against the {bias.lower()} thesis")

        if len(reasons) >= 2:
            retrace_tier = "HIGH"
            retrace_note = f"negative gamma with {' + '.join(reasons)} - real chop likely before the move confirms"
        elif len(reasons) == 1:
            retrace_tier = "ELEVATED"
            retrace_note = f"negative gamma with {reasons[0]} - some chop possible before confirmation"
        else:
            retrace_tier = "LOW"
            retrace_note = (
                "negative gamma but no early-session, vanna, or magnet-opposition flags - "
                "a clean expansion without testing lower levels first is plausible, though this "
                "regime still won't defend a reversal if one starts"
            )

    lines = [
        f"QUICK READ: {lvl.get('Market State', 'n/a')} - {bias} bias, "
        f"{lvl.get('Confidence', 'n/a')}% confidence ({lvl.get('Primary Regime', 'n/a')})",
    ]
    if key_level is not None:
        lines.append(
            f"Key level ({key_source}): {key_level['name']} @ {key_level['level']:,.1f} "
            f"(dominance {key_level['dominance_pct']}%) - most likely to get reacted on today"
        )
    if magnet_level is not None:
        dist = abs(magnet_level - spot)
        stance = "opposes" if magnet_opposes else "supports"
        lines.append(
            f"Magnet: {magnet_name} @ {magnet_level:,.1f} ({dist:,.1f} pts {magnet_direction}, "
            f"{magnet_strength} pull) - {stance} the {bias.lower()} thesis"
        )
    lines.append(f"Retracement Risk: {retrace_tier} - {retrace_note}")
    lines.append(f"Plan: {lvl.get('Entry Trigger', 'n/a')}")
    return "\n".join(lines)


def backquant_block(label, lvl, spot, expiry, odte=None, ndx_check=None):
    now = session_clock_str()
    d = dte(expiry)
    lo, hi = lvl.get("Expected Move Lo", spot), lvl.get("Expected Move Hi", spot)
    flow = lvl.get("Flow Confidence", {})
    chain_text = lvl.get("Chain Quality Summary", "")
    chain_text = f"Chain Quality: {chain_text}\n" if chain_text else ""
    if ndx_check and ndx_check.get("available"):
        ndx_text = (
            f"NDX Cross-Check: NDX ATM IV {ndx_check['ndx_atm_iv']}% vs QQQ-derived ATM IV "
            f"{ndx_check['qqq_atm_iv']}% (delta {ndx_check['delta_pp']:+.2f}pp) - {ndx_check['note']}\n"
        )
    elif ndx_check:
        ndx_text = f"NDX Cross-Check: {ndx_check.get('reason', 'unavailable')}\n"
    else:
        ndx_text = ""
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
        f"{item['name']}: ${item['level']:,.1f} - {item['verdict']} ({item['score']}%) "
        f"[dominance {item['dominance_pct']}% - {item['dominance_label']}]"
        for item in lvl.get("Level Strength", [])
    )
    if odte and odte.get("available"):
        odte_strength_lines = "\n".join(
            f"{item['name']}: ${item['level']:,.1f} - {item['verdict']} ({item['score']}%) "
            f"[dominance {item['dominance_pct']}%] - {item['note']}"
            for item in odte.get("strength", [])
        )
        odte_lines = (
            f"0DTE Levels (expiry {odte['expiry']}, {odte['hours_to_close']:.1f}h to close):\n"
            f"0DTE Call: ${odte['call_wall']:,.1f}\n"
            f"0DTE Put: ${odte['put_wall']:,.1f}\n"
            f"0DTE Level Strength / Hold Risk:\n"
            f"{odte_strength_lines}\n"
        )
    elif odte:
        odte_lines = f"0DTE Levels: unavailable - {odte.get('reason', 'no same-day chain')}\n"
    else:
        odte_lines = ""
    quick_read = build_quick_read(lvl, odte, spot)
    return (
f"{quick_read}\n"
f"---\n"
f"GEX + DEX + VANNA + CHARM Levels [{label}] - {now}\n"
f"Data Quality: {lvl.get('Quality Summary', 'n/a')}\n"
f"{chain_text}"
f"{ndx_text}"
f"{warnings_text}"
f"Market State: {lvl['Market State']} - {lvl['Confidence']}% confidence (alt scenario: {lvl['Alt Scenario']} @ {lvl['Alt Probability']}%)\n"
f"{hierarchy_lines}"
f"Regime: {lvl['Regime']} | Direction: {lvl['DEX Bias']}\n"
f"Vanna: {lvl['Vanna Note']} | Charm: {lvl['Charm Note']}\n"
f"Tactical Execution: {lvl['Tactical Execution']}\n"
f"Entry Trigger: {lvl['Entry Trigger']}\n"
f"---\n"
f"Level Strength / Hold Risk:\n"
f"{strength_lines}\n"
f"---\n"
f"Core GEX Levels:\n"
f"HVL (Gamma Anchor): ${lvl['HVL']:,.1f}\n"
f"GEX Flip: ${lvl['GEX Flip']:,.1f}\n"
f"Call Resistance: ${lvl['Call Wall']:,.1f}\n"
f"Put Support: ${lvl['Put Wall']:,.1f}\n"
f"{odte_lines}"
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
def gex_chart(filt, lvl, spot, label, odte=None):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#080c10")
    ax.set_facecolor("#080c10")
    x = filt["strike"]
    bw = (x.max()-x.min()) / max(len(x),1) * 0.75
    ax.bar(x, filt["call_gex"]/1e9, width=bw, color="#00e676", alpha=0.7, label="Call GEX")
    ax.bar(x, filt["put_gex"] /1e9, width=bw, color="#ff5252", alpha=0.7, label="Put GEX")
    ax.axhline(0, color="#2a3a2a", linewidth=0.8)
    lines = [
    (lvl["HVL"], "#ffd740", "-", 2.0, f"HVL {lvl['HVL']}"),
    (lvl["Call Wall"], "#00e676", "-", 2.0, f"CALL WALL {lvl['Call Wall']}"),
    (lvl["Put Wall"], "#ff5252", "-", 2.0, f"PUT WALL {lvl['Put Wall']}"),
    (lvl["GEX Flip"], "#00bcd4", "--", 1.5, f"GEX FLIP {lvl['GEX Flip']}"),
    (lvl["Vanna Flip"], "#ce93d8", "-.", 1.5, f"VANNA FLIP {lvl['Vanna Flip']}"),
    (lvl["Max Pain"], "#ffab40", ":", 1.2, f"MAX PAIN {lvl['Max Pain']}"),
    (spot, "#ffffff", "--", 1.0, f"SPOT {spot:.1f}"),
    ]
    if odte and odte.get("available"):
        lines.append((odte["call_wall"], "#00ff9c", ":", 1.6, f"0DTE CALL {odte['call_wall']}"))
        lines.append((odte["put_wall"], "#e040fb", ":", 1.6, f"0DTE PUT {odte['put_wall']}"))
    for val, col, ls, lw, lbl in lines:
        ax.axvline(val, color=col, linestyle=ls, linewidth=lw, label=lbl)

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
def render_level_card(lvl, spot, odte=None, ndx_check=None):
    reg_cls = "regime-blue" if "POSITIVE" in lvl["Regime"] else "regime-neg"
    dex_cls = "regime-pos" if "BULLISH" in lvl["DEX Bias"] else "regime-neg"
    van_cls = "regime-neg" if "BEARISH" in lvl["Vanna Regime"] else "regime-purple"
    chm_cls = "regime-orange" if "BULLISH" in lvl["Charm Flow"] else "regime-neg"
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
    if ndx_check and ndx_check.get("available"):
        ndx_cls = "regime-pos" if ndx_check.get("aligned") else "regime-orange"
        ndx_row = f"""<div class="level-row" style="padding:4px 0 8px 0;">
 <span class="level-label">NDX Cross-Check</span>
 <span class="{ndx_cls}" style="font-size:0.78rem;text-align:right;max-width:72%;">NDX IV {ndx_check['ndx_atm_iv']}% vs QQQ IV {ndx_check['qqq_atm_iv']}% ({ndx_check['delta_pp']:+.2f}pp) - {escape(ndx_check['note'])}</span>
 </div>"""
    elif ndx_check:
        ndx_row = f"""<div class="level-row" style="padding:4px 0 8px 0;">
 <span class="level-label">NDX Cross-Check</span>
 <span class="level-val" style="font-size:0.78rem;color:#5a7085;text-align:right;max-width:72%;">{escape(ndx_check.get('reason', 'unavailable'))}</span>
 </div>"""
    else:
        ndx_row = ""
    if odte and odte.get("available"):
        odte_strength_html = "".join(
            f"""<div class="level-row" style="align-items:flex-start;">
 <span class="level-label">{escape(item['name'])} {item['level']:,.1f}</span>
 <span class="{item['class']}" style="font-size:0.78rem;text-align:right;">
 {item['verdict']} {item['score']}%
 <span class="strength-bar"><span class="strength-fill" style="width:{item['score']}%;background:{'#4fc3f7' if item['score'] >= 70 else ('#ffab40' if item['score'] >= 45 else '#ff5252')};"></span></span>
 </span></div>
 <div class="level-row" style="padding:2px 0 6px 0;"><span class="level-label" style="font-size:0.7rem;color:#4a6070;">{escape(item['note'])} &nbsp;-&nbsp; dominance {item['dominance_pct']}%</span></div>"""
            for item in odte.get("strength", [])
        )
        odte_section = f"""
  <div class="section-divider"></div>
  <div class="section-header">0DTE LEVELS</div>
 <div class="level-row"><span class="level-label">0DTE Call</span>
 <span class="level-val lvl-green">{odte['call_wall']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">0DTE Put</span>
 <span class="level-val lvl-purple">{odte['put_wall']:,.1f}</span></div>
 <div class="level-row"><span class="level-label">0DTE ATM IV / Time</span>
 <span class="level-val" style="font-size:0.78rem;">{odte['atm_iv']*100:.1f}% &nbsp;|&nbsp; {odte['hours_to_close']:.1f}h to close</span></div>
 <div class="level-row" style="padding:6px 0 2px 0;"><span class="level-label" style="font-size:0.7rem;letter-spacing:0.5px;color:#4a6070;">0DTE HOLD / BREAK CONFIDENCE</span></div>
 {odte_strength_html}"""
    elif odte:
        odte_section = f"""
  <div class="section-divider"></div>
  <div class="section-header">0DTE LEVELS</div>
 <div class="level-row"><span class="level-label">Status</span>
 <span class="regime-orange" style="font-size:0.78rem;text-align:right;max-width:72%;">{escape(odte.get('reason', '0DTE chain unavailable'))}</span></div>"""
    else:
        odte_section = ""
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
 <div style="font-size:0.68rem;color:#5a7085;font-weight:normal;margin-top:2px;">dominance {item['dominance_pct']}% ({escape(item['dominance_label'])})</div>
 </span></div>"""
        for item in lvl.get("Level Strength", [])
    )
    flow = lvl.get("Flow Confidence", {})
    flow_rows = "".join(
        f"""<div class="level-row" style="align-items:flex-start;padding:5px 0;">
 <span class="level-label">{idx}. {escape(item['name'])} [{item['score']}%]</span>
 <span class="level-val" style="font-size:0.76rem;color:#8a9bb0;text-align:right;max-width:62%;">
 {escape(item['note_1'])}<br>{escape(item['note_2'])}
 </span></div>"""
        for idx, item in enumerate(flow.get("hierarchy", []), start=1)
    )

    quick_read_lines = build_quick_read(lvl, odte, spot).split("\n")
    quick_read_html = "".join(f'<div style="padding:2px 0;">{escape(line)}</div>' for line in quick_read_lines)

    st.markdown(f"""
 <div class="level-card" style="border:1px solid #ffd740;background:#1a1a0d;margin-bottom:10px;padding:12px 16px;">
  <div class="section-header" style="color:#ffd740;">QUICK READ</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;color:#e8e8e8;line-height:1.5;">
  {quick_read_html}
  </div>
 </div>

 <div class="level-card">

  <div class="section-divider"></div>
  <div class="section-header">MARKET VERDICT</div>
  <div class="level-row" style="padding:8px 0;">
  <span class="level-label">Market State</span>
  <span class="{lvl['Market Class']}" style="font-size:1rem;letter-spacing:0.5px;">{lvl['Market State']}</span>
  </div>
  <div class="level-row" style="padding:2px 0 6px 0;">
  <span class="level-label">Confidence</span>
  <span class="level-val" style="font-size:0.82rem;">{lvl['Confidence']}%</span>
  </div>
  <div class="level-row" style="padding:2px 0 6px 0;">
  <span class="level-label">Primary Regime / Bias</span>
  <span class="level-val" style="font-size:0.78rem;color:#8a9bb0;text-align:right;">{lvl['Primary Regime']} / {lvl['Directional Bias']}</span>
  </div>
  <div class="level-row" style="padding:2px 0 8px 0;">
  <span class="level-label">Alt Scenario</span>
  <span class="level-val" style="font-size:0.78rem;color:#8a9bb0;text-align:right;">{escape(lvl['Alt Scenario'])} ({lvl['Alt Probability']}%)</span>
  </div>
  {flow_rows}
  <div class="level-row" style="padding:7px 0 9px 0;border-top:1px solid #1a3050;margin-top:5px;">
  <span class="level-label">Tactical Execution</span>
  <span class="level-val" style="font-size:0.78rem;color:#c9d1d9;text-align:right;max-width:68%;">{escape(lvl['Tactical Execution'])}</span>
  </div>
 <div class="level-row" style="padding:4px 0 8px 0;">
 <span class="level-label">Data Quality</span>
  <span class="{quality_cls}" style="font-size:0.78rem;text-align:right;max-width:72%;">{escape(lvl.get('Quality Summary', 'n/a'))}</span>
  </div>
 <div class="level-row" style="padding:4px 0 8px 0;">
 <span class="level-label">Chain Quality</span>
 <span class="{chain_quality_cls}" style="font-size:0.78rem;text-align:right;max-width:72%;">{escape(lvl.get('Chain Quality Summary', 'n/a'))}</span>
 </div>
 {ndx_row}
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
  <div class="section-header">GEX - REGIME ENGINE</div>
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
 {odte_section}

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

    no_input_warn = ("No live open-price entered - confidence uses GEX/DEX/Vanna/Charm only"
                     if not d.get("has_open_price") else "Live price/IV momentum feeding into confidence")

    with st.expander("Signal Debug - hierarchy state", expanded=False):
        st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;line-height:2;color:#c9d1d9;
 background:#0d1117;border:1px solid #1e3a1e;border-radius:8px;padding:14px 18px;">
<b style="color:#ffd740">{no_input_warn}</b><br><br>

<b style="color:#4a6070">RAW INPUTS</b><br>
price_change = <b style="color:#{'00e676' if d.get('price_change',0) > 0 else ('ff5252' if d.get('price_change',0) < 0 else '8a9bb0')}">{d.get('price_change', 'n/a')}</b><br>
iv_change = <b style="color:#{'ff5252' if d.get('iv_change',0) > 0 else ('00e676' if d.get('iv_change',0) < 0 else '8a9bb0')}">{d.get('iv_change', 'n/a')}%</b><br><br>

<b style="color:#4a6070">MOMENTUM CONFIRMATION (confidence nudge only - never a separate verdict)</b><br>
has_open_price {tick(d.get('has_open_price'))}<br>
price_confirms_dex_bias {tick(d.get('price_confirms'))}<br>
iv_confirms_dex_bias {tick(d.get('iv_confirms'))}<br>
vanna_bullish {tick(d.get('vanna_bullish'))} (from Vanna regime label)<br>
vanna_bearish {tick(d.get('vanna_bearish'))}<br>
vanna_neutral {tick(d.get('vanna_neutral'))}<br><br>

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

if session_mode() == "curl_cffi":
    st.caption("Session: curl_cffi (hardened - Chrome TLS impersonation active)")
else:
    st.caption(
        "Session: fallback (curl_cffi not installed - add `curl_cffi` to requirements.txt "
        "for real anti-rate-limit protection; this mode is easy for Yahoo to block)"
    )

market_choice = st.radio("Market", ["US100", "US500"], horizontal=True)
cfd_spot = st.number_input(f"{market_choice} CFD spot", min_value=0.0, value=0.0, step=0.1, format="%.1f")
open_price_input = st.number_input(
    f"{market_choice} today's session open (optional - confirms live bias with real price action)",
    min_value=0.0, value=0.0, step=0.1, format="%.1f",
)

btn_col, cc_col = st.columns([4, 1])
with btn_col:
    run = st.button("Generate Levels")
with cc_col:
    if st.button("Clear Cache"):
        fetch_raw_chain.clear()
        fetch_raw_0dte_chain.clear()
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
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "too many" in msg:
                st.error(
                    "Yahoo is rate-limiting this connection right now. The app already retries "
                    "automatically with backoff, so if you're seeing this, the block has outlasted "
                    "that window - it needs a cooldown, not another click. Wait a few minutes before "
                    "trying again, and skip 'Clear Cache' unless you actually need a fresh chain (a new "
                    "expiry, or you suspect stale OI) - just updating the spot field re-hits the cache-"
                    "protected data for free and does not risk triggering this."
                )
            else:
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
    st.success(f"Levels ready - {session_clock_str()}")
    st.markdown(f'<span class="source-badge badge-src">{label} converted spot {item["index_spot"]:.1f}</span>', unsafe_allow_html=True)
    iv_dir = "down" if iv_pct < 0 else "up"
    st.markdown(f'ATM IV: **{item["atm_iv"]*100:.1f}%** | Open IV: **{open_iv*100:.1f}%** | IV Change: **{iv_dir} {abs(iv_pct):.1f}%**')
    render_debug_panel(lvl)
    odte = item.get("odte")
    ndx_check = item.get("ndx_check")
    render_level_card(lvl, item["index_spot"], odte, ndx_check)
    st.markdown("**BackQuant Paste Block:**")
    st.code(backquant_block(label, lvl, item["index_spot"], item["expiry"], odte=odte, ndx_check=ndx_check), language=None)
    st.pyplot(gex_chart(item["filt"], lvl, item["index_spot"], label, odte=odte))

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
