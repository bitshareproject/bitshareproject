# fx_trader_ev.py
"""
FX Trader EV — OANDA + Boxes + Event Trades + Basket Strength + Hierarchical Bayes
==================================================================================

NEW: BASKET STRENGTH
    For every trade we compute how strong the base and quote currencies
    were relative to the equal-weight FX basket at entry:

        base_strength   z-score of base ccy 20-bar return vs basket
        quote_strength  z-score of quote ccy 20-bar return vs basket

    Sign is aligned to trade direction:
        for a LONG  on XXX_YYY, strong XXX and weak YYY → positive
        for a SHORT on XXX_YYY, weak XXX and strong YYY → positive

    Bucketed into {weak, neutral, strong} for the cell chain.
    Sign-aligned so positive = tailwind for the trade direction.

FOUR SETUPS:
    range      traded inside the box (mean-revert)
    breakout   rode the box break (direction aligned)
    reversal   faded the break / caught a fakeout
    event      positioned around a high-impact release

ENTRY → SETUP MATCHING:
    Soft distribution over the four setups per trade, driven by box_state,
    direction, event_imminent, and the box features. Hard label = argmax.

PREDICTION:
    p_win_final  = Σ_s  p_setup_s  ·  p_win_s
    ev_bps_final = Σ_s  p_setup_s  ·  ev_bps_s
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Iterable

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

OANDA_HOST_PRACTICE = "https://api-fxpractice.oanda.com"
OANDA_HOST_LIVE     = "https://api-fxtrade.oanda.com"
OANDA_TOKEN = os.environ.get("OANDA_TOKEN", "YOUR_OANDA_TOKEN")
OANDA_ENV   = os.environ.get("OANDA_ENV", "practice").lower()
OANDA_HOST  = OANDA_HOST_LIVE if OANDA_ENV == "live" else OANDA_HOST_PRACTICE

FINNHUB_TOKEN = os.environ.get("FINNHUB_TOKEN", "")

DEFAULT_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF",
    "USD_CAD", "NZD_USD", "EUR_JPY", "GBP_JPY", "EUR_GBP",
    "AUD_JPY", "EUR_AUD", "EUR_CHF", "GBP_CHF", "AUD_NZD",
]

GRANULARITY_SPECS: Dict[str, Dict[str, Any]] = {
    "H1": {"years": 1.5, "halflife_days": 45.0,  "box_lookback": 48,
           "atr_lookback": 14, "vwap_window": 24, "vwap_slope_window": 6,
           "strength_lookback": 48},
    "H4": {"years": 3.0, "halflife_days": 90.0,  "box_lookback": 36,
           "atr_lookback": 14, "vwap_window": 20, "vwap_slope_window": 5,
           "strength_lookback": 36},
    "D":  {"years": 6.0, "halflife_days": 180.0, "box_lookback": 20,
           "atr_lookback": 14, "vwap_window": 20, "vwap_slope_window": 5,
           "strength_lookback": 20},
}

DEFAULT_GRANULARITIES = ["H4", "D"]

LIBRARY_PATH_DEFAULT   = "fx_trader_library.pkl"
LIBRARY_FORMAT_VERSION = 8

DEFAULT_HALFLIFE_DAYS     = 60.0
DEFAULT_PRIOR_STRENGTH    = 4.0
DEFAULT_CLUSTER_INFLATION = 2.0
DEFAULT_COST_BPS          = 1.5
DEFAULT_WILSON_Z          = 1.96
DEFAULT_MAX_RPS           = 5.0
DEFAULT_FETCH_WORKERS     = 4
HTTP_TIMEOUT              = 45
HTTP_RETRIES              = 3
HTTP_BACKOFF              = 1.6

DEFAULT_EVENT_WINDOW_MIN = 30
DEFAULT_MIN_EVENT_IMPACT = 3

# Thresholds for bucketing basket strength z-scores
STRENGTH_BUCKET_LOW  = -0.5
STRENGTH_BUCKET_HIGH =  0.5

LABEL_WIN     = "WIN"
LABEL_LOSS    = "LOSS"
LABEL_SCRATCH = "SCRATCH"
LABEL_REVERT  = "REVERT"
LABELS = (LABEL_WIN, LABEL_LOSS, LABEL_SCRATCH, LABEL_REVERT)

BOX_STATES = ("inside", "breakout_up", "breakout_dn", "fakeout_up", "fakeout_dn")

SETUP_RANGE    = "range"
SETUP_BREAKOUT = "breakout"
SETUP_REVERSAL = "reversal"
SETUP_EVENT    = "event"
SETUPS         = (SETUP_RANGE, SETUP_BREAKOUT, SETUP_REVERSAL, SETUP_EVENT)

EVENT_CCY_NONE   = "none"
EVENT_CCY_BASE   = "base"
EVENT_CCY_QUOTE  = "quote"
EVENT_CCY_BOTH   = "both"
EVENT_CCY_VALUES = (EVENT_CCY_NONE, EVENT_CCY_BASE, EVENT_CCY_QUOTE, EVENT_CCY_BOTH)

STRENGTH_BUCKETS = ("weak", "neutral", "strong")

TIME_BRACKETS = ("overnight", "asia", "london", "overlap", "ny", "late_ny")


# ============================================================
# SETUP MATCHING (soft)
# ============================================================

def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temp, 1e-9)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def setup_match_scores(box_state: str, direction: str,
                       event_imminent: bool,
                       brk_dist_atr: float = 0.0,
                       pos_in_box: float = 0.5,
                       vwap_dist_atr: float = 0.0,
                       touch_balance: float = 0.0,
                       base_strength: float = 0.0,
                       quote_strength: float = 0.0
                       ) -> Dict[str, float]:
    s_range = 0.0
    s_breakout = 0.0
    s_reversal = 0.0
    s_event = 0.0

    if box_state == "inside":
        s_range += 2.0
        s_breakout -= 0.5
        s_reversal -= 0.5
    elif box_state == "breakout_up":
        if direction == "L":
            s_breakout += 2.0
        else:
            s_reversal += 2.0
    elif box_state == "breakout_dn":
        if direction == "S":
            s_breakout += 2.0
        else:
            s_reversal += 2.0
    elif box_state == "fakeout_up":
        if direction == "S":
            s_reversal += 2.5
        else:
            s_breakout += 1.0
    elif box_state == "fakeout_dn":
        if direction == "L":
            s_reversal += 2.5
        else:
            s_breakout += 1.0

    if np.isfinite(brk_dist_atr):
        s_breakout += 0.8 * max(brk_dist_atr, 0.0)
        s_reversal -= 0.6 * max(brk_dist_atr, 0.0)

    if np.isfinite(pos_in_box):
        centrality = 1.0 - min(abs(pos_in_box - 0.5) * 2.0, 1.0)
        s_range += 1.0 * centrality
        s_breakout -= 0.5 * centrality
        s_reversal += 0.3 * centrality

    if np.isfinite(vwap_dist_atr):
        s_breakout += 0.3 * min(abs(vwap_dist_atr), 3.0)
        s_range    += 0.4 * max(0.0, 1.0 - abs(vwap_dist_atr))

    if np.isfinite(touch_balance):
        s_range    += 0.5 * (1.0 - abs(touch_balance))
        s_breakout += 0.4 * abs(touch_balance)

    # Basket strength: aligned tailwind favors continuation (breakout),
    # adverse (negative) strength favors mean-revert (reversal/range)
    if np.isfinite(base_strength) and np.isfinite(quote_strength):
        combo = 0.5 * (base_strength + quote_strength)   # sign-aligned
        s_breakout += 0.6 * combo
        s_reversal -= 0.4 * combo
        s_range    -= 0.2 * combo

    if event_imminent:
        s_event += 3.0
        s_range -= 0.5
        s_breakout -= 0.5
        s_reversal -= 0.5

    return {
        SETUP_RANGE:    s_range,
        SETUP_BREAKOUT: s_breakout,
        SETUP_REVERSAL: s_reversal,
        SETUP_EVENT:    s_event,
    }


def setup_match_distribution(box_state: str, direction: str,
                             event_imminent: bool,
                             brk_dist_atr: float = 0.0,
                             pos_in_box: float = 0.5,
                             vwap_dist_atr: float = 0.0,
                             touch_balance: float = 0.0,
                             base_strength: float = 0.0,
                             quote_strength: float = 0.0,
                             temperature: float = 1.0
                             ) -> Dict[str, float]:
    scores = setup_match_scores(box_state, direction, event_imminent,
                                brk_dist_atr, pos_in_box,
                                vwap_dist_atr, touch_balance,
                                base_strength, quote_strength)
    keys = list(scores.keys())
    vec = np.array([scores[k] for k in keys])
    p = _softmax(vec, temp=temperature)
    return {k: float(pi) for k, pi in zip(keys, p)}


def argmax_setup(dist: Dict[str, float]) -> str:
    return max(dist, key=dist.get)


# ============================================================
# BASKET STRENGTH
# ============================================================

def strength_bucket(z: float) -> str:
    if not np.isfinite(z):
        return "neutral"
    if z <= STRENGTH_BUCKET_LOW:
        return "weak"
    if z >= STRENGTH_BUCKET_HIGH:
        return "strong"
    return "neutral"


def compute_basket_strength(bars_by_pair_gran: Dict[Tuple[str, str], pd.DataFrame],
                            strength_lookback: int
                            ) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Add base_strength_raw / quote_strength_raw (z-scored basket-relative
    returns) to each pair's bar DataFrame.

    For each bar t and each currency c:
        ret_c(t)  = log(close_pair(t)) - log(close_pair(t - L))  for pairs
                    where c is the base
                    minus log-return for pairs where c is the quote
                    (sign-flipped so positive = currency c strengthened)
        strength_c(t) = (ret_c(t) - mean(ret_c over time)) / std(ret_c)

    Then per pair (XXX_YYY):
        base_strength_raw  = strength_XXX(t)
        quote_strength_raw = -strength_YYY(t)   (a strong quote is bearish)
    """
    # Build a per-currency log-return series from every pair we have
    # Key: currency, Value: DataFrame indexed by datetime with the
    # direction-adjusted log return of that currency.
    per_ccy_ret: Dict[str, pd.Series] = {}

    for (pair, g), df in bars_by_pair_gran.items():
        if df is None or df.empty:
            continue
        try:
            base, quote = pair.split("_")
        except Exception:
            continue
        logp = np.log(df["close"].values.astype(float))
        ret = np.full_like(logp, np.nan)
        L = max(int(strength_lookback), 2)
        ret[L:] = logp[L:] - logp[:-L]
        s = pd.Series(ret, index=pd.to_datetime(df["datetime"]))
        # base ccy: +ret ; quote ccy: -ret
        per_ccy_ret.setdefault(base, []).append(s)
        per_ccy_ret.setdefault(quote, []).append(-s)

    # Average across pairs for each currency
    ccy_avg: Dict[str, pd.Series] = {}
    for ccy, lst in per_ccy_ret.items():
        if not lst:
            continue
        cat = pd.concat(lst, axis=1)
        ccy_avg[ccy] = cat.mean(axis=1, skipna=True)

    # Z-score each currency's series
    ccy_z: Dict[str, pd.Series] = {}
    for ccy, s in ccy_avg.items():
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if not np.isfinite(sd) or sd <= 0:
            ccy_z[ccy] = s * 0.0
        else:
            ccy_z[ccy] = (s - mu) / sd

    # Attach to each pair's DataFrame
    for (pair, g), df in bars_by_pair_gran.items():
        if df is None or df.empty:
            continue
        try:
            base, quote = pair.split("_")
        except Exception:
            continue
        idx = pd.to_datetime(df["datetime"])
        bs = ccy_z.get(base, pd.Series(0.0, index=idx)).reindex(idx).values
        qs = ccy_z.get(quote, pd.Series(0.0, index=idx)).reindex(idx).values
        df = df.copy()
        df["base_strength_raw"]  = bs
        df["quote_strength_raw"] = -qs   # strong quote is bearish for the pair
        df["strength_combo_raw"] = 0.5 * (df["base_strength_raw"]
                                          + df["quote_strength_raw"])
        bars_by_pair_gran[(pair, g)] = df

    return bars_by_pair_gran


# ============================================================
# THREAD-SAFE HTTP + RATE LIMIT
# ============================================================

_thread_local  = threading.local()
_rate_lock     = threading.Lock()
_last_req_time = [0.0]


def _get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://",  adapter)
        _thread_local.session = s
    return s


def _rate_limit(max_rps: float) -> None:
    if max_rps <= 0:
        return
    min_gap = 1.0 / max_rps
    with _rate_lock:
        now = time.monotonic()
        delta = now - _last_req_time[0]
        if delta < min_gap:
            time.sleep(min_gap - delta)
        _last_req_time[0] = time.monotonic()


def _http_get_json(url: str, params: Optional[dict],
                   headers: Optional[dict],
                   max_rps: float) -> Optional[dict]:
    session = _get_session()
    for attempt in range(HTTP_RETRIES):
        _rate_limit(max_rps)
        try:
            r = session.get(url, params=params or {},
                            headers=headers or {}, timeout=HTTP_TIMEOUT)
            if r.status_code == 429:
                time.sleep(HTTP_BACKOFF ** (attempt + 1)); continue
            if r.status_code >= 500:
                time.sleep(HTTP_BACKOFF ** (attempt + 1)); continue
            if r.status_code != 200:
                return None
            return r.json()
        except (requests.RequestException, ValueError):
            time.sleep(HTTP_BACKOFF ** (attempt + 1))
    return None


# ============================================================
# OANDA CANDLE FETCH
# ============================================================

def fetch_candles_oanda(instrument: str, granularity: str, years: float,
                        max_rps: float) -> pd.DataFrame:
    try:
        end   = datetime.utcnow()
        start = end - timedelta(days=int(years * 365.25) + 10)
        url   = f"{OANDA_HOST}/v3/instruments/{instrument}/candles"
        headers = {"Authorization": f"Bearer {OANDA_TOKEN}",
                   "Accept": "application/json",
                   "Content-Type": "application/json"}
        params: Optional[dict] = {
            "granularity": granularity,
            "price": "M",
            "from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        rows: List[dict] = []
        pages = 0
        while url:
            data = _http_get_json(url, params, headers, max_rps)
            if not isinstance(data, dict) or "candles" not in data:
                if pages == 0 and isinstance(data, dict):
                    print(f"    [{instrument} {granularity}] "
                          f"status={data.get('errorCode')} "
                          f"msg={data.get('errorMessage')}")
                break
            for c in data["candles"]:
                if not c.get("complete", False):
                    continue
                m = c["mid"]
                rows.append({
                    "datetime": pd.to_datetime(c["time"]).tz_localize(None),
                    "open":  float(m["o"]), "high": float(m["h"]),
                    "low":   float(m["l"]), "close": float(m["c"]),
                    "volume": int(c.get("volume", 0)),
                })
            pages += 1
            nxt = data.get("next")
            if nxt:
                url, params = nxt, {}
            else:
                url = None
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows)
                  .sort_values("datetime")
                  .drop_duplicates("datetime")
                  .reset_index(drop=True))
    except Exception as e:
        print(f"    [{instrument} {granularity}] fetch error: "
              f"{type(e).__name__}: {e}")
        return pd.DataFrame()


def fetch_many_pairs(pairs: List[str], granularities: List[str],
                     years_by_gran: Dict[str, float],
                     workers: int, max_rps: float
                     ) -> Dict[Tuple[str, str], pd.DataFrame]:
    cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    jobs = [(p, g) for p in pairs for g in granularities]
    if not jobs:
        return cache
    workers = max(1, min(workers, len(jobs)))
    print(f"  Launching {workers} fetch threads for {len(jobs)} "
          f"(pair, granularity) jobs (rate cap ~{max_rps:.1f} req/sec) ...")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="oanda") as ex:
        futs = {ex.submit(fetch_candles_oanda, p, g, years_by_gran[g], max_rps): (p, g)
                for (p, g) in jobs}
        done = 0
        for fut in as_completed(futs):
            pair, g = futs[fut]
            done += 1
            try:
                df = fut.result()
            except Exception as e:
                print(f"  [{done:>3}/{len(jobs)}] {pair:<8} {g:<3} "
                      f"FAILED {type(e).__name__}: {e}")
                continue
            if df is None or df.empty:
                print(f"  [{done:>3}/{len(jobs)}] {pair:<8} {g:<3}  empty")
                continue
            cache[(pair, g)] = df
            print(f"  [{done:>3}/{len(jobs)}] {pair:<8} {g:<3}  "
                  f"bars={len(df):>5}  "
                  f"{df['datetime'].iloc[0]:%Y-%m-%d} → "
                  f"{df['datetime'].iloc[-1]:%Y-%m-%d}")
    print(f"  Fetch complete: {len(cache)}/{len(jobs)} pairs in "
          f"{time.monotonic() - t0:.2f}s")
    return cache


# ============================================================
# ECONOMIC CALENDAR (unchanged from v7)
# ============================================================

@dataclass
class EconEvent:
    ts: pd.Timestamp
    currency: str
    impact: int
    name: str
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    source: str = ""


def _safe_float(x):
    try:
        if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def fetch_calendar_oanda_forexlabs(start: datetime, end: datetime,
                                   max_rps: float) -> List[EconEvent]:
    api = "https://api.oanda.com/forex-labs/calendar"
    headers = {"Accept": "application/json"}
    if OANDA_TOKEN and OANDA_TOKEN != "YOUR_OANDA_TOKEN":
        headers["Authorization"] = f"Bearer {OANDA_TOKEN}"
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    data = _http_get_json(api, params, headers, max_rps)
    out: List[EconEvent] = []
    if not isinstance(data, dict):
        return out
    for e in (data.get("events") or data.get("data") or []):
        try:
            ts = pd.to_datetime(e["timestamp"]).tz_localize(None)
            out.append(EconEvent(
                ts=ts,
                currency=str(e.get("currency", "")).upper(),
                impact=int(e.get("impact", 0)),
                name=str(e.get("event") or e.get("name") or ""),
                actual=_safe_float(e.get("actual")),
                forecast=_safe_float(e.get("forecast")),
                previous=_safe_float(e.get("previous")),
                source="oanda",
            ))
        except Exception:
            continue
    return out


def fetch_calendar_finnhub(start: datetime, end: datetime,
                           max_rps: float) -> List[EconEvent]:
    if not FINNHUB_TOKEN:
        return []
    url = "https://finnhub.io/api/v1/calendar/economic"
    headers = {"X-Finnhub-Token": FINNHUB_TOKEN,
               "Accept": "application/json"}
    params = {"from": start.strftime("%Y-%m-%d"),
              "to":   end.strftime("%Y-%m-%d")}
    data = _http_get_json(url, params, headers, max_rps)
    out: List[EconEvent] = []
    if not isinstance(data, dict):
        return out
    for e in (data.get("economicCalendar") or []):
        try:
            ts = pd.to_datetime(e["time"]).tz_localize(None)
            out.append(EconEvent(
                ts=ts,
                currency=str(e.get("currency", "")).upper(),
                impact=int(e.get("impact", 0)),
                name=str(e.get("event", "")),
                actual=_safe_float(e.get("actual")),
                forecast=_safe_float(e.get("estimate")),
                previous=_safe_float(e.get("prev")),
                source="finnhub",
            ))
        except Exception:
            continue
    return out


def load_calendar_from_csv(path: str) -> List[EconEvent]:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    out: List[EconEvent] = []
    for _, r in df.iterrows():
        try:
            out.append(EconEvent(
                ts=pd.to_datetime(r["timestamp"]).tz_localize(None),
                currency=str(r["currency"]).upper(),
                impact=int(r["impact"]),
                name=str(r.get("event", "")),
                actual=_safe_float(r.get("actual")),
                forecast=_safe_float(r.get("forecast")),
                previous=_safe_float(r.get("previous")),
                source="csv",
            ))
        except Exception:
            continue
    return out


def get_calendar(start: datetime, end: datetime, max_rps: float,
                 events_csv: Optional[str] = None) -> pd.DataFrame:
    events: List[EconEvent] = []
    if events_csv and os.path.isfile(events_csv):
        events = load_calendar_from_csv(events_csv)
        print(f"  Calendar: loaded {len(events)} events from {events_csv}")
    else:
        if OANDA_TOKEN and OANDA_TOKEN != "YOUR_OANDA_TOKEN":
            events = fetch_calendar_oanda_forexlabs(start, end, max_rps)
            print(f"  Calendar: {len(events)} events from OANDA ForexLabs")
        if not events and FINNHUB_TOKEN:
            events = fetch_calendar_finnhub(start, end, max_rps)
            print(f"  Calendar: {len(events)} events from Finnhub")
        if not events:
            print("  Calendar: no source available — event_imminent will "
                  "default to False for all trades.")

    if not events:
        return pd.DataFrame(columns=["ts", "currency", "impact", "name",
                                     "actual", "forecast", "previous", "source"])
    df = pd.DataFrame([asdict(e) for e in events])
    return (df.sort_values("ts")
              .drop_duplicates(["ts", "currency", "name"])
              .reset_index(drop=True))


def annotate_events(trades: pd.DataFrame, calendar: pd.DataFrame,
                    window_min: int, min_impact: int) -> pd.DataFrame:
    out = trades.copy()
    out["event_imminent"] = False
    out["event_currency"] = EVENT_CCY_NONE
    out["event_name"]     = ""
    out["event_impact"]   = 0
    out["event_minutes"]  = 0

    if calendar is None or len(calendar) == 0:
        return out

    hi = calendar[calendar["impact"] >= int(min_impact)].copy()
    if hi.empty:
        return out
    hi_ts = pd.to_datetime(hi["ts"]).values
    win = pd.Timedelta(minutes=int(window_min))

    for idx, row in out.iterrows():
        entry_ts = pd.Timestamp(row["entry_ts"])
        pair = row["pair"]
        try:
            base, quote = pair.split("_")
        except Exception:
            base = quote = ""
        deltas = np.abs(pd.to_datetime(hi_ts) - entry_ts)
        within = deltas <= win
        if not within.any():
            continue
        j = int(np.argmin(deltas[within]))
        cand = hi.iloc[np.where(within)[0][j]]
        cur = str(cand["currency"]).upper()

        cur_match = EVENT_CCY_NONE
        if cur == base and cur == quote:
            cur_match = EVENT_CCY_BOTH
        elif cur == base:
            cur_match = EVENT_CCY_BASE
        elif cur == quote:
            cur_match = EVENT_CCY_QUOTE
        else:
            if cur == "USD":
                cur_match = (EVENT_CCY_BASE if base == "USD"
                             else EVENT_CCY_QUOTE if quote == "USD"
                             else EVENT_CCY_NONE)

        out.at[idx, "event_imminent"] = True
        out.at[idx, "event_currency"] = cur_match
        out.at[idx, "event_name"]     = str(cand["name"])
        out.at[idx, "event_impact"]   = int(cand["impact"])
        out.at[idx, "event_minutes"]  = int(
            (entry_ts - pd.Timestamp(cand["ts"])).total_seconds() // 60)

    return out


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def wilson_interval(k: float, n: float, z: float = DEFAULT_WILSON_Z,
                    inflation: float = DEFAULT_CLUSTER_INFLATION
                    ) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    var_scale = max(inflation, 1.0)
    d = 1.0 + z * z * var_scale / n
    c = p + z * z * var_scale / (2.0 * n)
    h = z * np.sqrt(max(p * (1.0 - p) * var_scale / n
                        + (z * z) * (var_scale ** 2) / (4.0 * n * n), 0.0))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s1 = w.sum(); s2 = (w * w).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def time_weights(ages_days: np.ndarray, halflife_days: float) -> np.ndarray:
    hl = max(float(halflife_days), 1e-6)
    return np.exp(-np.log(2.0) * np.asarray(ages_days, dtype=float) / hl)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      q: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    v = v[ok]; w = w[ok]
    order = np.argsort(v)
    v = v[order]; w = w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    idx = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(v[min(idx, len(v) - 1)])


# ============================================================
# CONFIG DATACLASS
# ============================================================

@dataclass
class TraderConfig:
    halflife_days: float = DEFAULT_HALFLIFE_DAYS
    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    cluster_inflation: float = DEFAULT_CLUSTER_INFLATION
    cost_bps: float = DEFAULT_COST_BPS
    wilson_z: float = DEFAULT_WILSON_Z

    box_lookback: int = 36
    atr_lookback: int = 14
    vwap_window: int = 20
    vwap_slope_window: int = 5
    min_breakout_atr: float = 0.10
    max_height_atr: float = 4.0
    touch_tolerance_atr: float = 0.35
    wick_tol_frac: float = 0.10

    strength_lookback: int = 36

    event_window_min: int = DEFAULT_EVENT_WINDOW_MIN
    min_event_impact: int = DEFAULT_MIN_EVENT_IMPACT

    setup_temperature: float = 1.0

    vol_pct_bins: Tuple[float, float] = (1.0/3.0, 2.0/3.0)

    min_n_fine: int = 8
    min_n_trader: int = 20
    min_n_pair: int = 30
    min_n_ccy: int = 30
    min_n_global: int = 100

    sessions: Tuple[str, ...] = ("ASIA", "LN", "NY", "TOK", "SYD")
    regimes: Tuple[str, ...] = ("trend", "range", "vol", "quiet")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def rolling_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    prev = np.empty_like(c); prev[0] = c[0]; prev[1:] = c[:-1]
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    return pd.Series(tr).rolling(period, min_periods=2).mean().values


def rolling_vwap(df: pd.DataFrame, window: int) -> np.ndarray:
    tp = (df["high"].values + df["low"].values + df["close"].values) / 3.0
    vol = df["volume"].values.astype(np.float64)
    pv = pd.Series(tp * vol)
    v  = pd.Series(vol)
    num = pv.rolling(window, min_periods=max(2, window // 4)).sum().values
    den = v.rolling(window,  min_periods=max(2, window // 4)).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def realized_vol_pct(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    r = df["close"].pct_change()
    vol = r.rolling(lookback, min_periods=5).std().values
    n = len(vol)
    finite = np.isfinite(vol)
    if finite.sum() < 20:
        return np.full(n, 0.5)
    s = pd.Series(vol)
    def _rk(x):
        return (x.iloc[-1] >= x).mean() if x.notna().any() else np.nan
    pct = s.rolling(lookback * 4, min_periods=lookback).apply(_rk, raw=False).values
    return np.where(np.isfinite(pct), pct, 0.5)


def vol_bucket(vol_pct: float, cfg: TraderConfig) -> str:
    if not np.isfinite(vol_pct):
        return "mid"
    lo, hi = cfg.vol_pct_bins
    if vol_pct <= lo:  return "low"
    if vol_pct >= hi:  return "high"
    return "mid"


def tenure_bucket(y: float) -> str:
    if not np.isfinite(y): return "mid"
    if y < 2.0:  return "junior"
    if y < 7.0:  return "mid"
    return "senior"


def session_from_hour(h: int) -> str:
    if 0 <= h < 7:   return "TOK"
    if 7 <= h < 12:  return "LN"
    if 12 <= h < 22: return "NY"
    return "SYD"


def time_bracket_from_hour(h: int) -> str:
    if h >= 22 or h < 2:  return "overnight"
    if h < 7:             return "asia"
    if h < 12:            return "london"
    if h < 15:            return "overlap"
    if h < 20:            return "ny"
    return "late_ny"


def regime_from_features(vol_pct: float, trend_score: float) -> str:
    if not np.isfinite(vol_pct):
        return "range"
    if vol_pct >= 2.0/3.0 and abs(trend_score) < 0.3:
        return "vol"
    if abs(trend_score) >= 0.5:
        return "trend"
    if vol_pct <= 1.0/3.0:
        return "quiet"
    return "range"


# ============================================================
# BOX DETECTION
# ============================================================

def detect_box_states(df: pd.DataFrame, cfg: TraderConfig) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n < cfg.box_lookback + 5:
        out["box_state"] = "inside"
        out["upper"] = np.nan; out["lower"] = np.nan
        out["height_atr"] = np.nan; out["brk_dist_atr"] = np.nan
        out["pos_in_box"] = np.nan; out["vwap_dist_atr"] = np.nan
        out["vwap_slope_atr"] = np.nan; out["vwap_above"] = 0
        out["touch_balance"] = 0.0; out["atr"] = np.nan
        out["vol_pct"] = 0.5
        if "base_strength_raw" not in out.columns:
            out["base_strength_raw"]  = 0.0
            out["quote_strength_raw"] = 0.0
            out["strength_combo_raw"] = 0.0
        return out

    atr = rolling_atr(out, cfg.atr_lookback)
    vwap = rolling_vwap(out, cfg.vwap_window)
    out["atr"] = atr

    out["upper"] = out["high"].rolling(cfg.box_lookback).quantile(0.95)
    out["lower"] = out["low"].rolling(cfg.box_lookback).quantile(0.05)

    height = out["upper"] - out["lower"]
    out["height_atr"] = height / atr

    c = out["close"].values
    u = out["upper"].values
    l = out["lower"].values
    a = atr

    brk_up_dist = (c - u) / np.where(a > 0, a, np.nan)
    brk_dn_dist = (l - c) / np.where(a > 0, a, np.nan)
    inside   = (c >= l) & (c <= u)
    break_up = (c > u) & (brk_up_dist >= cfg.min_breakout_atr)
    break_dn = (c < l) & (brk_dn_dist >= cfg.min_breakout_atr)

    prev_up = np.roll(break_up, 1); prev_up[0] = False
    prev_dn = np.roll(break_dn, 1); prev_dn[0] = False
    fake_up = prev_up & inside
    fake_dn = prev_dn & inside

    state = np.full(n, "inside", dtype=object)
    state[break_up] = "breakout_up"
    state[break_dn] = "breakout_dn"
    state[fake_up]  = "fakeout_up"
    state[fake_dn]  = "fakeout_dn"
    out["box_state"] = state

    brk_dist = np.where(break_up, brk_up_dist,
                np.where(break_dn, brk_dn_dist, 0.0))
    out["brk_dist_atr"] = brk_dist

    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(height.values > 0, (c - l) / height.values, 0.5)
    out["pos_in_box"] = np.clip(pos, -0.5, 1.5)

    vwap_dist = (c - vwap) / np.where(a > 0, a, np.nan)
    out["vwap_dist_atr"] = np.where(np.isfinite(vwap_dist), vwap_dist, 0.0)

    k = max(1, cfg.vwap_slope_window)
    vw_prev = np.roll(vwap, k); vw_prev[:k] = np.nan
    slope = (vwap - vw_prev) / (k * np.where(a > 0, a, np.nan))
    out["vwap_slope_atr"] = np.where(np.isfinite(slope), slope, 0.0)
    out["vwap_above"] = (c > vwap).astype(int)

    tol = cfg.touch_tolerance_atr * a
    upper_touch = (out["high"].values >= (u - tol)).astype(float)
    lower_touch = (out["low"].values  <= (l + tol)).astype(float)
    w = cfg.box_lookback
    tu = pd.Series(upper_touch).rolling(w, min_periods=1).sum().values
    tl = pd.Series(lower_touch).rolling(w, min_periods=1).sum().values
    tot = np.maximum(tu + tl, 1.0)
    out["touch_balance"] = (tl - tu) / tot

    out["vol_pct"] = realized_vol_pct(out)

    if "base_strength_raw" not in out.columns:
        out["base_strength_raw"]  = 0.0
        out["quote_strength_raw"] = 0.0
        out["strength_combo_raw"] = 0.0

    return out


# ============================================================
# ANNOTATE TRADES WITH BOX STATE + STRENGTH
# ============================================================

def annotate_trades_with_boxes(trades: pd.DataFrame,
                               bars_by_pair_gran: Dict[Tuple[str, str], pd.DataFrame]
                               ) -> pd.DataFrame:
    trades = trades.copy()
    out_rows: List[Dict[str, Any]] = []
    for _, tr in trades.iterrows():
        row = tr.to_dict()
        pair = row["pair"]
        direction = row.get("direction", "L")
        entry_ts = pd.Timestamp(row["entry_ts"])

        chosen = None
        chosen_gran = None
        for g in ("H1", "H4", "D"):
            key = (pair, g)
            bars = bars_by_pair_gran.get(key)
            if bars is None or bars.empty:
                continue
            idx = bars["datetime"].searchsorted(entry_ts, side="right") - 1
            if idx < 0:
                continue
            chosen = bars.iloc[idx]
            chosen_gran = g
            break

        if chosen is None:
            row.update({
                "granularity": None,
                "box_state": "inside",
                "box_height_atr": np.nan, "box_upper": np.nan,
                "box_lower": np.nan, "brk_dist_atr": 0.0,
                "vwap_dist_atr": 0.0, "vwap_slope_atr": 0.0,
                "vwap_above": 0, "pos_in_box": 0.5,
                "touch_balance": 0.0, "atr_at_entry": np.nan,
                "vol_pct_at_entry": 0.5,
                "base_strength_raw": 0.0, "quote_strength_raw": 0.0,
                "strength_combo_raw": 0.0,
                "base_strength": 0.0, "quote_strength": 0.0,
                "base_strength_bucket": "neutral",
                "quote_strength_bucket": "neutral",
            })
            out_rows.append(row)
            continue

        bs_raw = float(chosen.get("base_strength_raw", 0.0) or 0.0)
        qs_raw = float(chosen.get("quote_strength_raw", 0.0) or 0.0)
        # sign-align to trade direction
        if direction == "S":
            bs = -bs_raw
            qs = -qs_raw
        else:
            bs = bs_raw
            qs = qs_raw

        row.update({
            "granularity": chosen_gran,
            "box_state": chosen["box_state"],
            "box_height_atr": float(chosen["height_atr"])
                              if pd.notna(chosen["height_atr"]) else np.nan,
            "box_upper": float(chosen["upper"])
                         if pd.notna(chosen["upper"]) else np.nan,
            "box_lower": float(chosen["lower"])
                         if pd.notna(chosen["lower"]) else np.nan,
            "brk_dist_atr": float(chosen["brk_dist_atr"]),
            "vwap_dist_atr": float(chosen["vwap_dist_atr"]),
            "vwap_slope_atr": float(chosen["vwap_slope_atr"]),
            "vwap_above": int(chosen["vwap_above"]),
            "pos_in_box": float(chosen["pos_in_box"]),
            "touch_balance": float(chosen["touch_balance"]),
            "atr_at_entry": float(chosen["atr"])
                            if pd.notna(chosen["atr"]) else np.nan,
            "vol_pct_at_entry": float(chosen["vol_pct"])
                                if pd.notna(chosen["vol_pct"]) else 0.5,
            "base_strength_raw":  bs_raw,
            "quote_strength_raw": qs_raw,
            "strength_combo_raw": 0.5 * (bs_raw + qs_raw),
            "base_strength":  bs,
            "quote_strength": qs,
            "base_strength_bucket":  strength_bucket(bs),
            "quote_strength_bucket": strength_bucket(qs),
        })
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# ============================================================
# SYNTHETIC TRADES (fallback if no --trades-csv)
# ============================================================

def make_synthetic_trades(pairs: List[str], n_traders: int = 40,
                          trades_per_trader: int = 200,
                          seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    styles  = ["discretionary", "systematic", "mean-revert",
               "breakout", "market-make", "event-driven"]
    regimes  = ["trend", "range", "vol", "quiet"]
    events   = ["NFP", "CPI", "FOMC", "ECB", "BoJ", "BoE", "RBA"]
    box_states = ["inside", "breakout_up", "breakout_dn",
                  "fakeout_up", "fakeout_dn"]

    trader_skill = {f"T{i+1:03d}": rng.normal(0, 0.55) for i in range(n_traders)}
    pair_edge    = {p: rng.normal(0, 0.25) for p in pairs}
    regime_edge  = {"trend": 0.18, "range": 0.10, "vol": -0.05, "quiet": 0.02}
    session_edge = {"ASIA": -0.02, "LN": 0.10, "NY": 0.06,
                    "TOK": -0.04, "SYD": -0.08}
    setup_edge   = {"range": -0.02, "breakout": 0.12,
                    "reversal": 0.05, "event": -0.08}

    rows: List[Dict[str, Any]] = []
    t0 = datetime(2022, 1, 1); span = 900
    for tid, skill in trader_skill.items():
        style = rng.choice(styles)
        tenure = float(rng.uniform(0.5, 15.0))
        home_pairs = rng.choice(pairs, size=int(rng.integers(2, 6)),
                                replace=False).tolist()
        home_brackets = rng.choice(TIME_BRACKETS,
                                   size=int(rng.integers(1, 3)),
                                   replace=False).tolist()
        for k in range(trades_per_trader):
            pair = rng.choice(home_pairs) if rng.random() < 0.7 else rng.choice(pairs)

            if rng.random() < 0.65:
                br = rng.choice(home_brackets)
                hour_ranges = {
                    "overnight": [22, 23, 0, 1],
                    "asia":      list(range(2, 7)),
                    "london":    list(range(7, 12)),
                    "overlap":   list(range(12, 15)),
                    "ny":        list(range(15, 20)),
                    "late_ny":   list(range(20, 22)),
                }
                hour = int(rng.choice(hour_ranges[br]))
            else:
                hour = int(rng.integers(0, 24))

            session = session_from_hour(hour)
            regime  = rng.choice(regimes)
            direction = rng.choice(["L", "S"])
            vol_pct = float(np.clip(rng.beta(2, 2), 0.01, 0.99))
            age_days = float(rng.uniform(0, span))
            entry_dt = t0 + timedelta(days=age_days, hours=hour)
            hold_h = float(rng.lognormal(mean=np.log(4.0), sigma=0.9))
            exit_dt = entry_dt + timedelta(hours=hold_h)

            box_state = rng.choice(box_states)
            event_imminent = bool(rng.random() < 0.15)
            if event_imminent:
                event_name = rng.choice(events)
                event_currency = rng.choice(
                    [EVENT_CCY_BASE, EVENT_CCY_QUOTE, EVENT_CCY_BOTH])
            else:
                event_name = ""
                event_currency = EVENT_CCY_NONE

            # Synthetic basket strength
            bs_raw = float(rng.normal(0, 1.0))
            qs_raw = float(rng.normal(0, 1.0))
            if direction == "S":
                bs = -bs_raw; qs = -qs_raw
            else:
                bs = bs_raw; qs = qs_raw

            dist = setup_match_distribution(
                box_state, direction, event_imminent,
                brk_dist_atr=float(rng.uniform(0.1, 0.9)),
                pos_in_box=float(rng.uniform(0.05, 0.95)),
                vwap_dist_atr=float(rng.normal(0, 0.6)),
                touch_balance=float(rng.uniform(-0.5, 0.5)),
                base_strength=bs,
                quote_strength=qs,
                temperature=0.7,
            )
            setup = argmax_setup(dist)

            base, quote = pair.split("_")
            logit = (skill + pair_edge[pair] + regime_edge[regime]
                     + session_edge[session] + setup_edge[setup]
                     + 0.20 * bs + 0.20 * qs        # strength tailwind helps
                     + 0.05 * np.log1p(tenure)
                     + rng.normal(0, 0.9))
            p_win = 1.0 / (1.0 + np.exp(-logit))
            if rng.random() < p_win:
                pnl = float(abs(rng.normal(28.0, 20.0)) + 4.0)
                label = LABEL_WIN
            else:
                pnl = float(-abs(rng.normal(22.0, 16.0)) - 3.0)
                if abs(pnl) < 8.0 and rng.random() < 0.5:
                    pnl = float(rng.normal(0, 3.0)); label = LABEL_SCRATCH
                else:
                    label = LABEL_LOSS
            if label == LABEL_LOSS and rng.random() < 0.15:
                label = LABEL_REVERT

            entry_px = 1.0 + rng.normal(0, 0.05)
            exit_px  = entry_px * (1.0 + (pnl / 1e4) * (1 if direction == "L" else -1))
            rows.append({
                "trade_id":  f"{tid}-{k:04d}",
                "trader_id": tid, "style": style,
                "tenure_years": tenure,
                "pair": pair, "base_ccy": base, "quote_ccy": quote,
                "direction": direction,
                "session": session, "regime": regime,
                "vol_pct": vol_pct,
                "box_state": box_state,
                "setup": setup,
                "event_imminent": event_imminent,
                "event_currency": event_currency,
                "event_name": event_name,
                "event_impact": (3 if event_imminent else 0),
                "event_minutes": (int(rng.integers(-30, 31))
                                  if event_imminent else 0),
                "base_strength_raw": bs_raw, "quote_strength_raw": qs_raw,
                "strength_combo_raw": 0.5 * (bs_raw + qs_raw),
                "base_strength": bs, "quote_strength": qs,
                "base_strength_bucket":  strength_bucket(bs),
                "quote_strength_bucket": strength_bucket(qs),
                "entry_ts": entry_dt, "exit_ts": exit_dt,
                "entry_px": entry_px, "exit_px": exit_px,
                "size_lots": float(rng.uniform(0.5, 5.0)),
                "pnl_bps": pnl, "label": label,
            })
    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"])
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"])
    return df


# ============================================================
# ENRICH TRADES
# ============================================================

def enrich_trades(df: pd.DataFrame, cfg: TraderConfig) -> pd.DataFrame:
    out = df.copy()

    if "base_ccy" not in out.columns or out["base_ccy"].isna().any():
        out["base_ccy"]  = out["pair"].str.split("_").str[0]
        out["quote_ccy"] = out["pair"].str.split("_").str[1]

    if "box_state" not in out.columns:
        out["box_state"] = "inside"
    if "event_imminent" not in out.columns:
        out["event_imminent"] = False
    if "event_currency" not in out.columns:
        out["event_currency"] = EVENT_CCY_NONE

    # default strength if missing
    if "base_strength" not in out.columns:
        out["base_strength"] = 0.0
    if "quote_strength" not in out.columns:
        out["quote_strength"] = 0.0
    if "base_strength_bucket" not in out.columns:
        out["base_strength_bucket"] = out["base_strength"].apply(strength_bucket)
    if "quote_strength_bucket" not in out.columns:
        out["quote_strength_bucket"] = out["quote_strength"].apply(strength_bucket)

    out["entry_ts"] = pd.to_datetime(out["entry_ts"])
    out["entry_hour"]   = out["entry_ts"].dt.hour
    out["time_bracket"] = out["entry_hour"].apply(time_bracket_from_hour)
    if "session" not in out.columns:
        out["session"] = out["entry_hour"].apply(session_from_hour)

    dists = [
        setup_match_distribution(
            bs, d, bool(ev),
            brk_dist_atr=float(brk) if np.isfinite(brk) else 0.0,
            pos_in_box=float(pib) if np.isfinite(pib) else 0.5,
            vwap_dist_atr=float(vd) if np.isfinite(vd) else 0.0,
            touch_balance=float(tb) if np.isfinite(tb) else 0.0,
            base_strength=float(bst) if np.isfinite(bst) else 0.0,
            quote_strength=float(qst) if np.isfinite(qst) else 0.0,
            temperature=cfg.setup_temperature,
        )
        for bs, d, ev, brk, pib, vd, tb, bst, qst in zip(
            out["box_state"].values,
            out["direction"].values,
            out["event_imminent"].values,
            out.get("brk_dist_atr", pd.Series([0.0]*len(out))).values,
            out.get("pos_in_box",   pd.Series([0.5]*len(out))).values,
            out.get("vwap_dist_atr",pd.Series([0.0]*len(out))).values,
            out.get("touch_balance",pd.Series([0.0]*len(out))).values,
            out["base_strength"].values,
            out["quote_strength"].values,
        )
    ]
    for s in SETUPS:
        out[f"p_setup_{s}"] = [d[s] for d in dists]
    out["setup"] = [argmax_setup(d) for d in dists]

    if "vol_pct_at_entry" in out.columns and out["vol_pct_at_entry"].notna().any():
        out["vol_bucket"] = out["vol_pct_at_entry"].apply(
            lambda v: vol_bucket(v, cfg))
    else:
        out["vol_bucket"] = out["vol_pct"].apply(lambda v: vol_bucket(v, cfg))

    out["tenure_bucket"] = out["tenure_years"].apply(tenure_bucket)

    out["is_win"]      = (out["label"] == LABEL_WIN).astype(int)
    out["is_scratch"]  = (out["label"] == LABEL_SCRATCH).astype(int)
    out["is_loss"]     = out["label"].isin([LABEL_LOSS, LABEL_REVERT]).astype(int)
    out["is_revert"]   = (out["label"] == LABEL_REVERT).astype(int)
    out["event_trade"] = out["event_imminent"].astype(int)
    out["net_pnl_bps"] = out["pnl_bps"] - cfg.cost_bps

    for s in SETUPS:
        out[f"is_{s}"] = (out["setup"] == s).astype(int)
    return out


# ============================================================
# CELL CHAIN (now includes basket strength)
# ============================================================

CELL_CHAIN: List[Tuple[str, ...]] = [
    ("trader_id", "pair", "regime", "session", "vol_bucket",
     "box_state", "event_currency",
     "base_strength_bucket", "quote_strength_bucket"),
    ("trader_id", "pair", "regime", "event_currency",
     "base_strength_bucket", "quote_strength_bucket"),
    ("trader_id", "pair", "event_currency",
     "base_strength_bucket", "quote_strength_bucket"),
    ("trader_id", "regime", "event_currency",
     "base_strength_bucket", "quote_strength_bucket"),
    ("trader_id", "time_bracket", "event_currency",
     "base_strength_bucket"),
    ("trader_id", "event_currency", "base_strength_bucket"),
    ("trader_id", "base_strength_bucket"),
    ("trader_id",),
    ("style", "time_bracket", "regime", "base_strength_bucket"),
    ("style", "regime"),
    ("pair", "regime", "event_currency",
     "base_strength_bucket", "quote_strength_bucket"),
    ("pair", "event_currency", "base_strength_bucket"),
    ("pair", "base_strength_bucket"),
    ("base_ccy", "event_currency", "base_strength_bucket"),
    ("quote_ccy", "event_currency", "quote_strength_bucket"),
    ("base_ccy",),
    ("quote_ccy",),
    ("regime", "event_currency"),
    ("event_currency",),
    ("time_bracket",),
    ("regime",),
    ("base_strength_bucket",),
    ("quote_strength_bucket",),
    (),
]


def buildable_cells(df: pd.DataFrame) -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    n = len(df)
    for cell in CELL_CHAIN:
        if len(cell) == 0:
            out.append(cell); continue
        card = 1
        for col in cell:
            card *= max(int(df[col].nunique(dropna=True)), 1)
        if card <= max(n // 2, 4):
            out.append(cell)
    return out


# ============================================================
# CELL AGGREGATION
# ============================================================

@dataclass
class CellStats:
    setup: str = ""
    n: int = 0
    n_eff: float = 0.0
    p_win: float = np.nan
    p_win_lo: float = np.nan
    p_win_hi: float = np.nan
    p_loss: float = np.nan
    p_scratch: float = np.nan
    p_revert: float = np.nan
    ev_bps: float = np.nan
    ev_median_bps: float = np.nan
    avg_win_bps: float = np.nan
    avg_loss_bps: float = np.nan
    payoff_ratio: float = np.nan
    profit_factor: float = np.nan
    ref_time: Optional[datetime] = None


def aggregate_cell(sub: pd.DataFrame, setup: str, ref_time: datetime,
                   cfg: TraderConfig) -> CellStats:
    s = CellStats(setup=setup)
    if sub is None or len(sub) == 0:
        return s

    ages = (ref_time - sub["exit_ts"]).dt.total_seconds().values / 86400.0
    ages = np.clip(ages, 0.0, None)
    w = time_weights(ages, cfg.halflife_days)
    n_eff = effective_sample_size(w)
    s.n = int(len(sub))
    s.n_eff = float(n_eff)
    s.ref_time = ref_time

    wsum = w.sum()
    if wsum <= 0:
        return s

    is_win     = sub["is_win"].values.astype(float)
    is_loss    = sub["is_loss"].values.astype(float)
    is_scratch = sub["is_scratch"].values.astype(float)
    is_revert  = sub["is_revert"].values.astype(float)
    pnl        = sub["net_pnl_bps"].values.astype(float)

    k_win     = float((is_win * w).sum())
    k_loss    = float((is_loss * w).sum())
    k_scratch = float((is_scratch * w).sum())
    k_revert  = float((is_revert * w).sum())

    s.p_win     = k_win / wsum * 100.0
    s.p_loss    = k_loss / wsum * 100.0
    s.p_scratch = k_scratch / wsum * 100.0
    s.p_revert  = k_revert / wsum * 100.0

    lo, hi = wilson_interval(k_win, max(n_eff, 1e-9),
                             cfg.wilson_z, cfg.cluster_inflation)
    s.p_win_lo = lo * 100.0
    s.p_win_hi = hi * 100.0

    s.ev_bps        = weighted_mean(pnl, w)
    s.ev_median_bps = weighted_quantile(pnl, w, 0.50)

    win_mask  = is_win.astype(bool)
    loss_mask = is_loss.astype(bool)
    s.avg_win_bps  = weighted_mean(pnl[win_mask],  w[win_mask])  if win_mask.any()  else np.nan
    s.avg_loss_bps = weighted_mean(pnl[loss_mask], w[loss_mask]) if loss_mask.any() else np.nan

    if (np.isfinite(s.avg_win_bps) and np.isfinite(s.avg_loss_bps)
            and s.avg_loss_bps != 0):
        s.payoff_ratio = s.avg_win_bps / abs(s.avg_loss_bps)
        gross_win  = float((pnl[win_mask]  * w[win_mask]).sum())
        gross_loss = float(abs((pnl[loss_mask] * w[loss_mask]).sum()))
        s.profit_factor = gross_win / gross_loss if gross_loss > 0 else np.nan
    return s


# ============================================================
# HIERARCHICAL POOLING
# ============================================================

def pool_chain(cells_by_key: Dict[Tuple, CellStats],
               chain: List[Tuple[str, ...]],
               key: Tuple,
               key_index: Dict[str, int],
               prior_strength: float) -> Dict[str, float]:
    global_stats = cells_by_key.get(((), ()))
    if global_stats is None or not np.isfinite(global_stats.p_win):
        p_win, n_eff = 50.0, 0.0
        ev, p_loss, p_scratch = 0.0, 50.0, 0.0
        avg_win, avg_loss = np.nan, np.nan
    else:
        p_win, n_eff = global_stats.p_win, global_stats.n_eff
        ev, p_loss, p_scratch = (global_stats.ev_bps,
                                 global_stats.p_loss,
                                 global_stats.p_scratch)
        avg_win, avg_loss = global_stats.avg_win_bps, global_stats.avg_loss_bps

    for cell in chain:
        if len(cell) == 0:
            stats = global_stats
        else:
            cols = [key_index[c] for c in cell]
            cell_key = tuple(key[i] for i in cols)
            stats = cells_by_key.get((cell, cell_key))
        if stats is None or not np.isfinite(stats.p_win) or stats.n_eff <= 0:
            continue
        alpha = n_eff + prior_strength
        beta  = stats.n_eff
        p_win     = (p_win * alpha     + stats.p_win * beta)     / (alpha + beta)
        p_loss    = (p_loss * alpha    + stats.p_loss * beta)    / (alpha + beta)
        p_scratch = (p_scratch * alpha + stats.p_scratch * beta) / (alpha + beta)
        if np.isfinite(stats.ev_bps):
            ev = (ev * alpha + stats.ev_bps * beta) / (alpha + beta)
        if np.isfinite(stats.avg_win_bps) and np.isfinite(avg_win):
            avg_win = (avg_win * alpha + stats.avg_win_bps * beta) / (alpha + beta)
        if np.isfinite(stats.avg_loss_bps) and np.isfinite(avg_loss):
            avg_loss = (avg_loss * alpha + stats.avg_loss_bps * beta) / (alpha + beta)
        n_eff = alpha + beta

    payoff = (avg_win / abs(avg_loss)) if (
        np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0
    ) else np.nan

    lo, hi = wilson_interval(
        (p_win / 100.0) * max(n_eff, 1e-9), max(n_eff, 1e-9),
        DEFAULT_WILSON_Z, DEFAULT_CLUSTER_INFLATION)

    return {
        "p_win": p_win,
        "p_loss": p_loss,
        "p_scratch": p_scratch,
        "ev_bps": ev,
        "avg_win_bps": avg_win,
        "avg_loss_bps": avg_loss,
        "payoff_ratio": payoff,
        "n_eff": n_eff,
        "p_win_lo": lo * 100.0,
        "p_win_hi": hi * 100.0,
    }


# ============================================================
# LIBRARY — one sub-library per setup
# ============================================================

@dataclass
class SetupLibrary:
    setup: str
    cells: List[Tuple[str, ...]]
    key_index: Dict[str, int]
    stats: Dict[Tuple, CellStats]
    n_trades: int


@dataclass
class FxLibrary:
    version: int
    built_at: str
    ref_time: str
    by_setup: Dict[str, SetupLibrary]
    cfg: Dict[str, Any]
    n_trades: int
    n_traders: int
    n_pairs: int


KEY_ORDER = [
    "trade_id", "trader_id", "style", "tenure_years", "tenure_bucket",
    "pair", "base_ccy", "quote_ccy", "direction",
    "session", "time_bracket", "regime", "vol_pct", "vol_bucket",
    "box_state", "setup", "event_imminent", "event_currency",
    "base_strength", "quote_strength",
    "base_strength_bucket", "quote_strength_bucket",
]


def _key_index() -> Dict[str, int]:
    return {c: i for i, c in enumerate(KEY_ORDER)}


def build_setup_library(df: pd.DataFrame, setup: str,
                        ref_time: datetime,
                        cfg: TraderConfig) -> SetupLibrary:
    sub = df[df["setup"] == setup].copy()
    cells = buildable_cells(sub) if len(sub) > 0 else [()]
    stats: Dict[Tuple, CellStats] = {}

    for cell in cells:
        if len(cell) == 0:
            stats[((), ())] = aggregate_cell(sub, setup, ref_time, cfg)
            continue
        grouper = cell[0] if len(cell) == 1 else list(cell)
        try:
            grouped = sub.groupby(grouper, dropna=False)
        except KeyError:
            continue
        for key, chunk in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            stats[(cell, key)] = aggregate_cell(chunk, setup, ref_time, cfg)

    return SetupLibrary(
        setup=setup,
        cells=cells,
        key_index=_key_index(),
        stats=stats,
        n_trades=int(len(sub)),
    )


def build_library(df: pd.DataFrame, cfg: TraderConfig) -> FxLibrary:
    ref_time = df["exit_ts"].max().to_pydatetime()
    by_setup: Dict[str, SetupLibrary] = {}
    for setup in SETUPS:
        by_setup[setup] = build_setup_library(df, setup, ref_time, cfg)
    return FxLibrary(
        version=LIBRARY_FORMAT_VERSION,
        built_at=datetime.now().isoformat(timespec="seconds"),
        ref_time=ref_time.isoformat(),
        by_setup=by_setup,
        cfg=asdict(cfg),
        n_trades=int(len(df)),
        n_traders=int(df["trader_id"].nunique()),
        n_pairs=int(df["pair"].nunique()),
    )


# ============================================================
# PREDICTION — setup-weighted EV
# ============================================================

def trade_key(trade: Dict[str, Any]) -> Tuple:
    return tuple(trade.get(c) for c in KEY_ORDER)


def predict_trade(trade: Dict[str, Any], lib: FxLibrary,
                  cfg: Optional[TraderConfig] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = TraderConfig(**lib.cfg)
    key = trade_key(trade)
    out = dict(trade)

    dist = setup_match_distribution(
        trade.get("box_state", "inside"),
        trade.get("direction", "L"),
        bool(trade.get("event_imminent", False)),
        brk_dist_atr=float(trade.get("brk_dist_atr", 0.0) or 0.0),
        pos_in_box=float(trade.get("pos_in_box", 0.5) or 0.5),
        vwap_dist_atr=float(trade.get("vwap_dist_atr", 0.0) or 0.0),
        touch_balance=float(trade.get("touch_balance", 0.0) or 0.0),
        base_strength=float(trade.get("base_strength", 0.0) or 0.0),
        quote_strength=float(trade.get("quote_strength", 0.0) or 0.0),
        temperature=cfg.setup_temperature,
    )
    for s in SETUPS:
        out[f"p_setup_{s}"] = dist[s]
    out["chosen_setup"] = argmax_setup(dist)

    for setup in SETUPS:
        sub_lib = lib.by_setup.get(setup)
        if sub_lib is None or not sub_lib.stats:
            out[f"p_win_{setup}"]  = np.nan
            out[f"ev_bps_{setup}"] = np.nan
            out[f"n_eff_{setup}"]  = 0.0
            out[f"p_win_lo_{setup}"] = np.nan
            out[f"p_win_hi_{setup}"] = np.nan
            out[f"avg_win_bps_{setup}"]  = np.nan
            out[f"avg_loss_bps_{setup}"] = np.nan
            out[f"payoff_ratio_{setup}"] = np.nan
            continue
        pooled = pool_chain(sub_lib.stats, sub_lib.cells, key,
                            sub_lib.key_index, cfg.prior_strength)
        out[f"p_win_{setup}"]      = pooled["p_win"]
        out[f"p_win_lo_{setup}"]   = pooled["p_win_lo"]
        out[f"p_win_hi_{setup}"]   = pooled["p_win_hi"]
        out[f"ev_bps_{setup}"]     = pooled["ev_bps"]
        out[f"n_eff_{setup}"]      = pooled["n_eff"]
        out[f"avg_win_bps_{setup}"]  = pooled["avg_win_bps"]
        out[f"avg_loss_bps_{setup}"] = pooled["avg_loss_bps"]
        out[f"payoff_ratio_{setup}"] = pooled["payoff_ratio"]

    p_win_final = 0.0
    ev_final    = 0.0
    n_eff_final = 0.0
    w_used      = 0.0
    p_win_lo_w  = 0.0
    p_win_hi_w  = 0.0
    for s in SETUPS:
        w = dist.get(s, 0.0)
        pw = out.get(f"p_win_{s}", np.nan)
        ev = out.get(f"ev_bps_{s}", np.nan)
        ne = out.get(f"n_eff_{s}", 0.0) or 0.0
        lo = out.get(f"p_win_lo_{s}", np.nan)
        hi = out.get(f"p_win_hi_{s}", np.nan)
        if not np.isfinite(pw):
            continue
        p_win_final += w * pw
        if np.isfinite(ev):
            ev_final += w * ev
        n_eff_final += w * ne
        w_used += w
        if np.isfinite(lo): p_win_lo_w += w * lo
        if np.isfinite(hi): p_win_hi_w += w * hi

    if w_used > 0:
        p_win_final /= w_used
        ev_final    /= w_used
        n_eff_final /= w_used
        p_win_lo_w  /= w_used
        p_win_hi_w  /= w_used

    out["p_win_final"]  = p_win_final
    out["ev_bps_final"] = ev_final
    out["n_eff_final"]  = n_eff_final
    out["p_win_lo_final"] = p_win_lo_w
    out["p_win_hi_final"] = p_win_hi_w

    s_best = out["chosen_setup"]
    out["p_win_chosen"]  = out.get(f"p_win_{s_best}",  np.nan)
    out["ev_bps_chosen"] = out.get(f"ev_bps_{s_best}", np.nan)
    out["n_eff_chosen"]  = out.get(f"n_eff_{s_best}",  0.0)
    return out


def predict_many(trades: Iterable[Dict[str, Any]], lib: FxLibrary,
                 cfg: Optional[TraderConfig] = None) -> List[Dict[str, Any]]:
    return [predict_trade(t, lib, cfg) for t in trades]


# ============================================================
# CONSOLE REPORTING
# ============================================================

def _f(x, w=6, prec=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return " " * w
    return f"{x:>{w}.{prec}f}"


def print_library_summary(lib: FxLibrary) -> None:
    print()
    print("=" * 140)
    print(f"  FX TRADER LIBRARY  ·  built {lib.built_at}  ·  "
          f"ref_time={lib.ref_time}")
    print(f"  trades={lib.n_trades:,}  traders={lib.n_traders}  "
          f"pairs={lib.n_pairs}  "
          f"halflife={lib.cfg['halflife_days']}d  "
          f"prior_strength={lib.cfg['prior_strength']}  "
          f"cost={lib.cfg['cost_bps']}bps  "
          f"cluster_inflation={lib.cfg['cluster_inflation']}  "
          f"setup_temp={lib.cfg['setup_temperature']}")
    print("=" * 140)
    for setup in SETUPS:
        sub = lib.by_setup[setup]
        print()
        print(f"  --- SETUP = {setup.upper()}   "
              f"(n_trades={sub.n_trades}) ---")
        print(f"  {'cell':<55} {'#keys':>7} {'mean n':>8} {'mean n_eff':>11} "
              f"{'mean p_win':>11} {'mean EV':>9}")
        print("  " + "-" * 118)
        for cell in sub.cells:
            keys = [k for (c, k) in sub.stats.keys() if c == cell]
            if not keys:
                continue
            cs = [sub.stats[(cell, k)] for k in keys]
            ns = [c.n for c in cs]
            ne = [c.n_eff for c in cs]
            pw = [c.p_win for c in cs if np.isfinite(c.p_win)]
            ev = [c.ev_bps for c in cs if np.isfinite(c.ev_bps)]
            label = "×".join(cell) if cell else "(global)"
            print(f"  {label:<55} {len(keys):>7} "
                  f"{np.mean(ns):>8.1f} {np.mean(ne):>11.1f} "
                  f"{(np.mean(pw) if pw else float('nan')):>11.2f} "
                  f"{(np.mean(ev) if ev else float('nan')):>9.2f}")


def print_example_predictions(lib: FxLibrary, cfg: TraderConfig,
                              df: pd.DataFrame, n: int = 12) -> None:
    print()
    print("=" * 265)
    print("  EXAMPLE PREDICTIONS  ·  soft setup match + basket strength + "
          "setup-weighted P(win) / EV")
    print("=" * 265)
    print(f"  {'trade_id':<14} {'trader':<7} {'pair':<8} {'dir':<4} "
          f"{'box_state':<12} {'evt':<4} {'bStr':>6} {'qStr':>6} "
          f"{'pRng':>5} {'pBrk':>5} {'pRev':>5} {'pEvt':>5} "
          f"{'chosen':<9} "
          f"{'PwRng':>7} {'PwBrk':>7} {'PwRev':>7} {'PwEvt':>7} "
          f"{'EVrng':>7} {'EVbrk':>7} {'EVrev':>7} {'EVevt':>7} "
          f"{'PwFinal':>8} {'EVfinal':>8} {'n_eff':>7} "
          f"{'actual':>8} {'label':<8}")
    print("  " + "-" * 263)

    sample = df.sample(n=min(n, len(df)), random_state=1)
    for _, row in sample.iterrows():
        rec = row.to_dict()
        rec.setdefault("vol_bucket", vol_bucket(rec.get("vol_pct", 0.5), cfg))
        rec.setdefault("tenure_bucket", tenure_bucket(rec.get("tenure_years", 3.0)))
        rec.setdefault("time_bracket",
                       time_bracket_from_hour(int(rec.get("entry_hour", 12))))
        pred = predict_trade(rec, lib, cfg)
        print(f"  {str(pred.get('trade_id',''))[:13]:<14} "
              f"{str(pred.get('trader_id',''))[:6]:<7} "
              f"{str(pred.get('pair',''))[:7]:<8} "
              f"{str(pred.get('direction',''))[:3]:<4} "
              f"{str(pred.get('box_state',''))[:11]:<12} "
              f"{'Y' if pred.get('event_imminent') else 'n':<4} "
              f"{_f(pred.get('base_strength',0.0), 6, 2)} "
              f"{_f(pred.get('quote_strength',0.0), 6, 2)} "
              f"{_f(pred['p_setup_range'],    5, 2)} "
              f"{_f(pred['p_setup_breakout'], 5, 2)} "
              f"{_f(pred['p_setup_reversal'], 5, 2)} "
              f"{_f(pred['p_setup_event'],    5, 2)} "
              f"{pred['chosen_setup']:<9} "
              f"{_f(pred['p_win_range'],    7, 2)} "
              f"{_f(pred['p_win_breakout'], 7, 2)} "
              f"{_f(pred['p_win_reversal'], 7, 2)} "
              f"{_f(pred['p_win_event'],    7, 2)} "
              f"{_f(pred['ev_bps_range'],    7, 2)} "
              f"{_f(pred['ev_bps_breakout'], 7, 2)} "
              f"{_f(pred['ev_bps_reversal'], 7, 2)} "
              f"{_f(pred['ev_bps_event'],    7, 2)} "
              f"{_f(pred['p_win_final'], 8, 2)} "
              f"{_f(pred['ev_bps_final'], 8, 2)} "
              f"{_f(pred['n_eff_final'], 7, 1)} "
              f"{_f(pred.get('net_pnl_bps', pred.get('pnl_bps', np.nan)), 8, 2)} "
              f"{str(pred.get('label','')):<8}")


def print_recent_ranked(lib: FxLibrary, cfg: TraderConfig,
                        df: pd.DataFrame, n: int = 20) -> None:
    print()
    print("=" * 230)
    print("  MOST RECENT TRADES RANKED BY SETUP-WEIGHTED EV (net of cost)")
    print("=" * 230)
    recent = df.sort_values("exit_ts", ascending=False).head(n).copy()
    recent["vol_bucket"]    = recent.get("vol_pct_at_entry", recent["vol_pct"]).apply(
        lambda v: vol_bucket(v, cfg))
    recent["tenure_bucket"] = recent["tenure_years"].apply(tenure_bucket)
    if "time_bracket" not in recent.columns:
        recent["time_bracket"] = recent["entry_ts"].dt.hour.apply(
            time_bracket_from_hour)
    preds = [predict_trade(r, lib, cfg) for r in recent.to_dict("records")]
    preds.sort(key=lambda r: -r.get("ev_bps_final", -np.inf))
    print(f"  {'trade_id':<14} {'trader':<7} {'pair':<8} {'dir':<4} "
          f"{'box_state':<12} {'evt':<4} {'bStr':>6} {'qStr':>6} "
          f"{'pRng':>5} {'pBrk':>5} {'pRev':>5} {'pEvt':>5} "
          f"{'chosen':<9} {'PwFinal':>8} {'EVfinal':>8} {'n_eff':>7} "
          f"{'actual':>8} {'label':<8}")
    print("  " + "-" * 195)
    for p in preds:
        print(f"  {str(p.get('trade_id',''))[:13]:<14} "
              f"{str(p.get('trader_id',''))[:6]:<7} "
              f"{str(p.get('pair',''))[:7]:<8} "
              f"{str(p.get('direction',''))[:3]:<4} "
              f"{str(p.get('box_state',''))[:11]:<12} "
              f"{'Y' if p.get('event_imminent') else 'n':<4} "
              f"{_f(p.get('base_strength',0.0), 6, 2)} "
              f"{_f(p.get('quote_strength',0.0), 6, 2)} "
              f"{_f(p['p_setup_range'],    5, 2)} "
              f"{_f(p['p_setup_breakout'], 5, 2)} "
              f"{_f(p['p_setup_reversal'], 5, 2)} "
              f"{_f(p['p_setup_event'],    5, 2)} "
              f"{p['chosen_setup']:<9} "
              f"{_f(p['p_win_final'], 8, 2)} "
              f"{_f(p['ev_bps_final'], 8, 2)} "
              f"{_f(p['n_eff_final'], 7, 1)} "
              f"{_f(p.get('net_pnl_bps', p.get('pnl_bps', np.nan)), 8, 2)} "
              f"{str(p.get('label','')):<8}")


# ============================================================
# LIBRARY PERSISTENCE
# ============================================================

def save_library(path: str, lib: FxLibrary) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(lib, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved library -> {path}  ({sz:.2f} MB)")


def load_library(path: str) -> Optional[FxLibrary]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            lib = pickle.load(f)
    except Exception as e:
        print(f"  Failed to load library: {type(e).__name__}: {e}")
        return None
    if not isinstance(lib, FxLibrary):
        print("  Library payload is not an FxLibrary — ignoring.")
        return None
    if lib.version != LIBRARY_FORMAT_VERSION:
        print(f"  Library version mismatch: {lib.version} != "
              f"{LIBRARY_FORMAT_VERSION}")
        return None
    return lib


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="FX trader EV — soft setup matching + basket strength + "
                    "hierarchical Bayesian pooling.")

    ap.add_argument("--trades-csv", type=str, default=None)
    ap.add_argument("--events-csv", type=str, default=None)
    ap.add_argument("--library-path", type=str, default=LIBRARY_PATH_DEFAULT)
    ap.add_argument("--load-library", action="store_true")
    ap.add_argument("--rebuild", action="store_true")

    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--granularities", nargs="*", default=DEFAULT_GRANULARITIES,
                    choices=list(GRANULARITY_SPECS.keys()))

    ap.add_argument("--halflife-days", type=float, default=None)
    ap.add_argument("--prior-strength", type=float, default=DEFAULT_PRIOR_STRENGTH)
    ap.add_argument("--cluster-inflation", type=float,
                    default=DEFAULT_CLUSTER_INFLATION)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--event-window-min", type=int,
                    default=DEFAULT_EVENT_WINDOW_MIN)
    ap.add_argument("--min-event-impact", type=int,
                    default=DEFAULT_MIN_EVENT_IMPACT)
    ap.add_argument("--setup-temperature", type=float, default=1.0)
    ap.add_argument("--strength-lookback", type=int, default=None)

    ap.add_argument("--n-traders", type=int, default=40)
    ap.add_argument("--trades-per-trader", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--fetch-workers", type=int, default=DEFAULT_FETCH_WORKERS)
    ap.add_argument("--max-rps", type=float, default=DEFAULT_MAX_RPS)

    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--skip-fetch", action="store_true")

    args = ap.parse_args()

    # ---- 1. load / make trades ----
    if args.trades_csv:
        print(f"  Loading trades from {args.trades_csv}")
        df = pd.read_csv(args.trades_csv)
        df["entry_ts"] = pd.to_datetime(df["entry_ts"])
        df["exit_ts"]  = pd.to_datetime(df["exit_ts"])
    else:
        print(f"  Generating synthetic trades "
              f"({args.n_traders} traders × {args.trades_per_trader} trades)")
        df = make_synthetic_trades(
            pairs=args.pairs,
            n_traders=int(args.n_traders),
            trades_per_trader=int(args.trades_per_trader),
            seed=int(args.seed))

    # ---- 2. fetch OANDA candles + detect boxes + basket strength ----
    bars_by_pair_gran: Dict[Tuple[str, str], pd.DataFrame] = {}
    if not args.skip_fetch and OANDA_TOKEN and OANDA_TOKEN != "YOUR_OANDA_TOKEN":
        years_by_gran = {g: float(GRANULARITY_SPECS[g]["years"])
                         for g in args.granularities}
        print(f"  Fetching OANDA candles for {len(args.pairs)} pairs × "
              f"{len(args.granularities)} granularities ...")
        bars_by_pair_gran = fetch_many_pairs(
            args.pairs, args.granularities, years_by_gran,
            workers=int(args.fetch_workers), max_rps=float(args.max_rps))

        print("  Detecting boxes + basket strength ...")
        for g in args.granularities:
            base = asdict(TraderConfig())
            base.update({k: GRANULARITY_SPECS[g][k]
                         for k in ("box_lookback", "atr_lookback",
                                   "vwap_window", "vwap_slope_window",
                                   "strength_lookback")})
            cfg_g = TraderConfig(**base)
            for pair in args.pairs:
                key = (pair, g)
                if key not in bars_by_pair_gran:
                    continue
                bars_by_pair_gran[key] = detect_box_states(
                    bars_by_pair_gran[key], cfg_g)

            # Compute basket strength across all pairs at this granularity
            print(f"    [{g}] computing basket strength "
                  f"(lookback={GRANULARITY_SPECS[g]['strength_lookback']}) ...")
            bars_by_pair_gran = compute_basket_strength(
                bars_by_pair_gran,
                strength_lookback=int(GRANULARITY_SPECS[g]["strength_lookback"]))

        if bars_by_pair_gran:
            print("  Annotating trades with box state + strength at entry ...")
            df = annotate_trades_with_boxes(df, bars_by_pair_gran)
    else:
        if args.skip_fetch:
            print("  --skip-fetch: not pulling OANDA candles.")
        else:
            print("  OANDA_TOKEN not set — skipping OANDA fetch; using "
                  "synthetic box states + synthetic strength in the panel.")

    # ---- 3. economic calendar ----
    cal_start = df["entry_ts"].min().to_pydatetime() - timedelta(days=1)
    cal_end   = df["entry_ts"].max().to_pydatetime() + timedelta(days=1)
    print(f"  Pulling economic calendar {cal_start.date()} → {cal_end.date()} ...")
    calendar = get_calendar(cal_start, cal_end, float(args.max_rps),
                            events_csv=args.events_csv)

    if len(calendar) > 0:
        df = annotate_events(df, calendar,
                             window_min=int(args.event_window_min),
                             min_impact=int(args.min_event_impact))
        n_ev = int(df["event_imminent"].sum())
        print(f"  Event annotation: {n_ev:,} / {len(df):,} trades "
              f"({100.0 * n_ev / max(len(df), 1):.1f}%) flagged as "
              f"event trades (±{args.event_window_min} min, "
              f"impact ≥ {args.min_event_impact})")
    else:
        print("  No calendar available — event_imminent defaults to False.")

    # ---- 4. config + enrich ----
    base = asdict(TraderConfig())
    if args.halflife_days:
        base["halflife_days"] = float(args.halflife_days)
    if args.strength_lookback:
        base["strength_lookback"] = int(args.strength_lookback)
    base["prior_strength"]    = float(args.prior_strength)
    base["cluster_inflation"] = float(args.cluster_inflation)
    base["cost_bps"]          = float(args.cost_bps)
    base["event_window_min"]  = int(args.event_window_min)
    base["min_event_impact"]  = int(args.min_event_impact)
    base["setup_temperature"] = float(args.setup_temperature)
    cfg = TraderConfig(**base)

    df = enrich_trades(df, cfg)
    print(f"  Trades: {len(df):,}  | traders: {df['trader_id'].nunique()}  "
          f"| pairs: {df['pair'].nunique()}")
    print(f"  Setup mix (argmax): {df['setup'].value_counts().to_dict()}")
    print(f"  Time-bracket mix: {df['time_bracket'].value_counts().to_dict()}")
    print(f"  Base-strength mix: "
          f"{df['base_strength_bucket'].value_counts().to_dict()}")
    print(f"  Quote-strength mix: "
          f"{df['quote_strength_bucket'].value_counts().to_dict()}")
    print(f"  Window: {df['entry_ts'].min().date()} → "
          f"{df['entry_ts'].max().date()}")

    # ---- 5. build / load library ----
    if args.load_library and not args.rebuild:
        lib = load_library(args.library_path)
        if lib is None:
            print("  Library not found or invalid; building ...")
            lib = None
    else:
        lib = None

    if lib is None:
        t0 = time.monotonic()
        lib = build_library(df, cfg)
        print(f"  Library built in {time.monotonic() - t0:.2f}s")
        save_library(args.library_path, lib)
    else:
        print(f"  Loaded library from {args.library_path}")

    print_library_summary(lib)

    # ---- 6. example predictions ----
    print_example_predictions(lib, cfg, df, n=int(args.n_examples))

    # ---- 7. recent ranked ----
    print_recent_ranked(lib, cfg, df, n=int(args.top))

    print()
    print(f"Summary: trades={len(df):,} · traders={df['trader_id'].nunique()} · "
          f"pairs={df['pair'].nunique()} · "
          f"setups={list(SETUPS)} · "
          f"setup_temp={args.setup_temperature}")


if __name__ == "__main__":
    main()
