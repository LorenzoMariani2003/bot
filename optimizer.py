"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CRYPTO STRATEGY OPTIMIZER v1.1 — MULTI-WINDOW VALIDATION          ║
║                                                                              ║
║  Modalità di ricerca:                                                        ║
║    --random N    → campiona N combinazioni casuali (default: 300)            ║
║    --grid        → enumera tutte le combo (attenzione: può essere lento)     ║
║                                                                              ║
║  Validazione multi-finestra temporale:                                       ║
║    --windows K   → K finestre casuali dallo storico (default: 5)            ║
║    --win-min D   → durata minima finestra in giorni (default: 90)            ║
║    --win-max D   → durata massima finestra in giorni (default: days//2)     ║
║                                                                              ║
║    Le finestre sono generate UNA SOLA VOLTA e condivise da tutte le         ║
║    combinazioni, così ogni param-set è valutato su periodi di mercato        ║
║    diversi (trend, laterale, crash, rally) e non solo sugli ultimi N gg.    ║
║                                                                              ║
║  Metriche di ranking (punteggio composito):                                  ║
║    Sharpe ratio  40% · Total return  30% · Win rate  20% · PF  10%          ║
║    × consistency_factor (% finestre profittevoli) × stability_factor         ║
║                                                                              ║
║  Uso:                                                                        ║
║    python optimizer.py --random 500 --days 730 --windows 6                  ║
║    python optimizer.py --random 300 --days 1095 --windows 8 --win-min 60    ║
║    python optimizer.py --grid   --days 365  --interval 4h --windows 4       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
import time
import warnings
import hashlib
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETRI E SPAZIO DI RICERCA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamSet:
    """
    Ogni campo è un iperparametro ottimizzabile.
    SEARCH_SPACE definisce i valori da esplorare per ogni campo.
    """
    # ── EMA ──────────────────────────────────────────────────────────────────
    ema_fast:  int   = 21
    ema_slow:  int   = 55
    ema_trend: int   = 200    # 0 = disabilitato (rimuove il filtro di trend)

    # ── ADX ──────────────────────────────────────────────────────────────────
    adx_period: int   = 14
    adx_min:    float = 22.0  # 0 = disabilitato

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_period: int   = 14
    rsi_ob:     float = 70.0  # overbought: NON comprare sopra questo valore
    rsi_os:     float = 30.0  # oversold: bonus se sotto questo valore

    # ── ATR (stop/take profit) ────────────────────────────────────────────────
    atr_period:    int   = 14
    atr_stop_mult: float = 2.0   # SL = entry - mult × ATR
    atr_tp_mult:   float = 3.0   # TP = entry + mult × ATR  (R:R ≈ 1.5)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_period: int   = 20
    bb_std:    float = 2.0

    # ── Volume spike ─────────────────────────────────────────────────────────
    vol_mult: float = 1.5   # spike se volume > vol_mult × SMA_volume(20)

    # ── Score soglia di ingresso ──────────────────────────────────────────────
    signal_threshold: float = 55.0

    # ── Position sizing (% del capitale per ogni trade) ───────────────────────
    position_pct: float = 0.15

    # ── Commissioni simulate ─────────────────────────────────────────────────
    commission: float = 0.001   # 0.1% per lato (Binance BNB holder)

    # ──────────────────────────────────────────────────────────────────────────
    # SPAZIO DI RICERCA
    # ──────────────────────────────────────────────────────────────────────────
    SEARCH_SPACE: Dict[str, List[Any]] = None

    def __post_init__(self):
        if self.SEARCH_SPACE is None:
            self.SEARCH_SPACE = {
                "ema_fast":  [8, 12, 13, 21, 34],
                "ema_slow":  [21, 34, 50, 55, 89],
                "ema_trend": [0, 50, 100, 150, 200],

                "adx_period": [10, 14, 20],
                "adx_min":    [0, 15, 18, 22, 25],

                "rsi_period": [10, 14, 21],
                "rsi_ob":     [60, 65, 70, 75, 80],
                "rsi_os":     [20, 25, 30, 35],

                "atr_period":    [10, 14, 21],
                "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
                "atr_tp_mult":   [2.0, 3.0, 4.0, 5.0],

                "bb_period": [15, 20, 25],
                "bb_std":    [1.5, 2.0, 2.5],

                "vol_mult": [1.2, 1.5, 2.0, 2.5, 3.0],

                "signal_threshold": [30, 40, 50, 55, 60, 65, 70],

                "position_pct": [0.10, 0.15, 0.20, 0.25],

                "commission": [0.001],
            }


def random_param_set(base: ParamSet) -> ParamSet:
    """Campiona un ParamSet casuale dallo spazio di ricerca."""
    space = base.SEARCH_SPACE
    kwargs = {}
    for f in fields(base):
        if f.name == "SEARCH_SPACE":
            continue
        if f.name in space:
            kwargs[f.name] = random.choice(space[f.name])
        else:
            kwargs[f.name] = getattr(base, f.name)
    if kwargs["ema_fast"] >= kwargs["ema_slow"]:
        kwargs["ema_fast"] = max(4, kwargs["ema_slow"] - 5)
    if kwargs["atr_tp_mult"] <= kwargs["atr_stop_mult"]:
        kwargs["atr_tp_mult"] = kwargs["atr_stop_mult"] + 1.0
    return ParamSet(**kwargs)


def grid_param_sets(base: ParamSet) -> List[ParamSet]:
    """Genera tutte le combinazioni possibili."""
    space  = base.SEARCH_SPACE
    keys   = [f.name for f in fields(base) if f.name != "SEARCH_SPACE"
                                             and f.name in space]
    values = [space[k] for k in keys]
    result = []
    for combo in itertools.product(*values):
        kwargs = dict(zip(keys, combo))
        if kwargs.get("ema_fast", 1) >= kwargs.get("ema_slow", 2):
            continue
        if kwargs.get("atr_tp_mult", 3) <= kwargs.get("atr_stop_mult", 2):
            continue
        result.append(ParamSet(**kwargs))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FINESTRE TEMPORALI
# ══════════════════════════════════════════════════════════════════════════════

# Numero di candele per giorno in base all'intervallo
CANDLES_PER_DAY: Dict[str, float] = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
    "1h": 24,   "2h": 12,  "4h": 6,   "6h": 4,   "8h": 3,
    "12h": 2,   "1d": 1.0, "3d": 1/3, "1w": 1/7,
}


@dataclass
class TimeWindow:
    """
    Finestra temporale definita come slice (start_offset, end_offset)
    sull'array di candele del DataFrame.
    """
    idx:          int   # numero progressivo (0-based)
    start_offset: int   # indice di inizio (incluso)
    end_offset:   int   # indice di fine (escluso)
    n_candles:    int   # numero di candele nella finestra

    def label(self) -> str:
        return f"W{self.idx + 1}[{self.start_offset}:{self.end_offset}]"


def generate_windows(data_map:  Dict[str, pd.DataFrame],
                     n_windows: int,
                     min_days:  int,
                     max_days:  int,
                     interval:  str,
                     seed:      int) -> List[TimeWindow]:
    """Genera N finestre temporali casuali, uguali per tutte le combinazioni."""
    cpd = CANDLES_PER_DAY.get(interval, 1.0)

    valid_dfs = [df for df in data_map.values() if not df.empty]
    if not valid_dfs:
        raise ValueError("Nessun dato disponibile per generare le finestre.")
    min_len = min(len(df) for df in valid_dfs)

    min_candles = max(60, int(min_days * cpd))
    max_candles = min(min_len - 10, int(max_days * cpd))

    if max_candles <= min_candles:
        print(f"\n  ⚠️  Storico troppo corto per finestre di {min_days}–{max_days}d "
              f"con intervallo {interval} ({min_len} candele disponibili).")
        print(f"     Uso l'intero dataset come unica finestra.")
        return [TimeWindow(idx=0, start_offset=0,
                           end_offset=min_len, n_candles=min_len)]

    rng     = random.Random(seed)
    windows: List[TimeWindow] = []
    attempts = 0

    while len(windows) < n_windows and attempts < n_windows * 30:
        attempts += 1
        size      = rng.randint(min_candles, max_candles)
        max_start = min_len - size
        if max_start <= 0:
            continue
        start = rng.randint(0, max_start)
        windows.append(TimeWindow(
            idx          = len(windows),
            start_offset = start,
            end_offset   = start + size,
            n_candles    = size,
        ))

    if not windows:
        windows.append(TimeWindow(idx=0, start_offset=0,
                                  end_offset=min_len, n_candles=min_len))
    return windows


def print_windows(windows: List[TimeWindow],
                  data_map: Dict[str, pd.DataFrame],
                  interval: str):
    cpd = CANDLES_PER_DAY.get(interval, 1.0)
    ref_df = next(iter(data_map.values()), None)

    print(f"\n  Finestre temporali generate ({len(windows)}):")
    print(f"  {'─'*60}")
    for win in windows:
        candles_info = f"{win.n_candles} candele (~{win.n_candles / cpd:.0f}gg)"
        if ref_df is not None and win.end_offset <= len(ref_df):
            t_start = ref_df["open_time"].iloc[win.start_offset].strftime("%Y-%m-%d")
            t_end   = ref_df["open_time"].iloc[win.end_offset - 1].strftime("%Y-%m-%d")
            print(f"    W{win.idx + 1:>2}: {t_start} → {t_end}  ({candles_info})")
        else:
            print(f"    W{win.idx + 1:>2}: offset [{win.start_offset}:{win.end_offset}]  ({candles_info})")
    print(f"  {'─'*60}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CACHE
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = ".ohlcv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, interval: str, days: int) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}_{days}d.parquet")


def fetch_klines(symbol: str, interval: str, days: int,
                 force_download: bool = False) -> pd.DataFrame:
    cache_file = _cache_path(symbol, interval, days)
    if not force_download and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 3600:
            return pd.read_parquet(cache_file)

    print(f"  ↓ Download {symbol} [{interval}] {days}d ...", end="", flush=True)
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    url      = "https://api.binance.com/api/v3/klines"
    all_rows = []
    current  = start_dt

    while True:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "limit":     1000,
            "startTime": int(current.timestamp() * 1000),
            "endTime":   int(end_dt.timestamp() * 1000),
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            print(f" ERRORE: {e}")
            return pd.DataFrame()

        if not rows:
            break
        all_rows.extend(rows)
        last_ts  = pd.Timestamp(rows[-1][0], unit="ms")
        current  = last_ts + timedelta(milliseconds=1)
        if len(rows) < 1000 or last_ts >= end_dt:
            break
        time.sleep(0.1)

    if not all_rows:
        print(" [nessun dato]")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df.to_parquet(cache_file, index=False)
    print(f" {len(df)} candele OK")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FAST VECTORIZED BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss     = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    rs       = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_c = close.shift(1)
    tr     = pd.concat([high - low,
                        (high - prev_c).abs(),
                        (low  - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    high, low = df["high"], df["low"]
    atr       = _atr(df, period)

    up   = high.diff().clip(lower=0)
    down = (-low.diff()).clip(lower=0)
    up[up < down]     = 0
    down[down < up]   = 0

    alpha    = 1 / period
    di_plus  = 100 * up.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * down.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    dx       = (100 * (di_plus - di_minus).abs() /
                (di_plus + di_minus).replace(0, np.nan))
    adx_val  = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx_val, di_plus, di_minus


def compute_signals(df: pd.DataFrame, p: ParamSet) -> pd.DataFrame:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    ema_f  = _ema(close, p.ema_fast)
    ema_s  = _ema(close, p.ema_slow)

    if p.ema_trend > 0:
        ema_t     = _ema(close, p.ema_trend)
        trend_ok  = (close > ema_t).astype(float)
    else:
        trend_ok  = pd.Series(1.0, index=df.index)

    if p.adx_min > 0:
        adx_val, di_plus, di_minus = _adx(df, p.adx_period)
        adx_ok = ((adx_val > p.adx_min) & (di_plus > di_minus)).astype(float)
    else:
        adx_ok = pd.Series(1.0, index=df.index)

    rsi = _rsi(close, p.rsi_period)
    rsi_score = np.where(
        rsi < p.rsi_os, 15.0,
        np.where(rsi < p.rsi_ob,
                 15.0 * (p.rsi_ob - rsi) / (p.rsi_ob - p.rsi_os + 1e-9),
                 0.0))

    atr = _atr(df, p.atr_period)

    bb_mid = close.rolling(p.bb_period).mean()
    bb_ok  = (close < bb_mid).astype(float)

    vol_ma = volume.rolling(20).mean()
    vol_ok = (volume > vol_ma * p.vol_mult).astype(float)

    score = (
        trend_ok * 25.0 +
        (ema_f > ema_s).astype(float) * 20.0 +
        adx_ok  * 15.0 +
        rsi_score +
        bb_ok   * 10.0 +
        vol_ok  * 10.0 +
        2.5
    )

    buy_sig  = (score >= p.signal_threshold)
    sell_sig = ((ema_f < ema_s) &
                (ema_f.shift(1) >= ema_s.shift(1)))

    return pd.DataFrame({
        "close": close.values,
        "high":  high.values,
        "low":   low.values,
        "atr":   atr.values,
        "score": score.values,
        "buy":   buy_sig.values,
        "sell":  sell_sig.values,
    }, index=df.index)


@dataclass
class BacktestResult:
    symbol:        str
    n_trades:      int
    win_rate:      float
    total_return:  float
    profit_factor: float
    max_drawdown:  float
    sharpe:        float
    avg_duration:  float
    final_capital: float
    composite:     float


def simulate(sig: pd.DataFrame, p: ParamSet,
             initial_capital: float = 100.0,
             symbol: str = "?") -> BacktestResult:
    capital  = initial_capital
    position = None
    trades   = []
    equity   = [capital]

    arr_close = sig["close"].values
    arr_high  = sig["high"].values
    arr_low   = sig["low"].values
    arr_atr   = sig["atr"].values
    arr_buy   = sig["buy"].values
    arr_sell  = sig["sell"].values

    n      = len(sig)
    warmup = max(p.ema_trend if p.ema_trend > 0 else 0, p.ema_slow) + 5

    for i in range(warmup, n):
        price = arr_close[i]

        if position is not None:
            sl    = position["sl"]
            tp    = position["tp"]
            entry = position["entry"]
            qty   = position["qty"]

            exit_price = None
            if arr_low[i] <= sl:
                exit_price = sl
            elif arr_high[i] >= tp:
                exit_price = tp
            elif arr_sell[i]:
                exit_price = price

            if exit_price is not None:
                gross  = (exit_price - entry) * qty
                comm   = (exit_price * qty * p.commission +
                          entry      * qty * p.commission)
                pnl    = gross - comm
                capital += pnl
                trades.append({
                    "pnl":      pnl,
                    "pnl_pct":  pnl / position["invested"] * 100,
                    "duration": i - position["entry_idx"],
                })
                position = None

        if position is None and arr_buy[i] and capital >= 10:
            atr    = arr_atr[i]
            invest = capital * p.position_pct
            invest = min(invest, capital)
            if invest < 5:
                equity.append(capital)
                continue

            comm_buy = invest * p.commission
            qty      = (invest - comm_buy) / price
            sl       = price - p.atr_stop_mult * atr
            tp       = price + p.atr_tp_mult  * atr

            position = {
                "entry":     price,
                "sl":        sl,
                "tp":        tp,
                "qty":       qty,
                "invested":  invest,
                "entry_idx": i,
            }

        equity.append(capital)

    if position is not None:
        price  = arr_close[-1]
        pnl    = (price - position["entry"]) * position["qty"]
        capital += pnl
        trades.append({
            "pnl":      pnl,
            "pnl_pct":  pnl / position["invested"] * 100,
            "duration": n - 1 - position["entry_idx"],
        })
        equity[-1] = capital

    n_trades = len(trades)
    if n_trades == 0:
        return BacktestResult(symbol, 0, 0, 0, 0, 0, 0, 0, capital, -999)

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls > 0

    gross_profit  = pnls[wins].sum()
    gross_loss    = abs(pnls[~wins].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 9.99

    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate     = wins.sum() / n_trades

    eq   = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / (peak + 1e-9)
    max_dd = float(dd.max() * 100)

    eq_ret = np.diff(eq) / (eq[:-1] + 1e-9)
    if len(eq_ret) > 1 and eq_ret.std() > 1e-9:
        sharpe = float(eq_ret.mean() / eq_ret.std() * math.sqrt(252))
    else:
        sharpe = 0.0

    avg_dur = float(np.mean([t["duration"] for t in trades]))

    trade_bonus = min(1.0, n_trades / 15)
    composite   = (
        sharpe        * 0.40 +
        total_return  * 0.30 +
        win_rate      * 100  * 0.20 +
        min(profit_factor, 5.0) * 2 * 0.10
    ) * trade_bonus

    return BacktestResult(
        symbol        = symbol,
        n_trades      = n_trades,
        win_rate      = float(win_rate * 100),
        total_return  = float(total_return),
        profit_factor = float(min(profit_factor, 9.99)),
        max_drawdown  = float(max_dd),
        sharpe        = float(sharpe),
        avg_duration  = float(avg_dur),
        final_capital = float(capital),
        composite     = float(composite),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST TRIAL — MULTI-WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_trial(args: Tuple) -> Optional[dict]:
    p, data_map, windows, initial_capital, min_trades = args

    per_window: List[dict] = []

    for win in windows:
        win_results: List[BacktestResult] = []

        for symbol, df in data_map.items():
            if df.empty:
                continue
            start = win.start_offset
            end   = min(win.end_offset, len(df))
            if end - start < 60:
                continue
            df_win = df.iloc[start:end].reset_index(drop=True)
            try:
                sig = compute_signals(df_win, p)
                r   = simulate(sig, p, initial_capital, symbol)
                win_results.append(r)
            except Exception:
                continue

        if not win_results:
            per_window.append({"n_trades": 0, "win_rate": 0.0,
                                "total_return": 0.0, "profit_factor": 0.0,
                                "max_drawdown": 0.0, "sharpe": 0.0,
                                "composite": -999.0, "profitable": False})
            continue

        total_trades_win = sum(r.n_trades for r in win_results)

        if total_trades_win == 0:
            per_window.append({"n_trades": 0, "win_rate": 0.0,
                                "total_return": 0.0, "profit_factor": 0.0,
                                "max_drawdown": 0.0, "sharpe": 0.0,
                                "composite": -999.0, "profitable": False})
            continue

        def wavg(attr: str) -> float:
            weights = [max(r.n_trades, 1) for r in win_results]
            vals    = [getattr(r, attr) for r in win_results]
            return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

        comp = wavg("composite")
        per_window.append({
            "n_trades":      total_trades_win,
            "win_rate":      wavg("win_rate"),
            "total_return":  wavg("total_return"),
            "profit_factor": wavg("profit_factor"),
            "max_drawdown":  wavg("max_drawdown"),
            "sharpe":        wavg("sharpe"),
            "avg_duration":  wavg("avg_duration"),
            "composite":     comp,
            "profitable":    wavg("total_return") > 0,
        })

    active       = [w for w in per_window if w["n_trades"] > 0]
    total_trades = sum(w["n_trades"] for w in active)

    if total_trades < min_trades:
        return None

    n_win        = len(per_window)
    n_profitable = sum(1 for w in per_window if w.get("profitable", False))
    consistency  = n_profitable / n_win if n_win > 0 else 0.0

    def cross_avg(key: str) -> float:
        if not active:
            return 0.0
        weights = [max(w["n_trades"], 1) for w in active]
        vals    = [w.get(key, 0.0) for w in active]
        return sum(v * wt for v, wt in zip(vals, weights)) / sum(weights)

    mean_composite = cross_avg("composite")

    composites    = [w["composite"] for w in active if w["composite"] > -100]
    composite_std = float(np.std(composites)) if len(composites) > 1 else 0.0

    consistency_factor = 0.5 + 0.5 * consistency
    stability_factor = 1.0 / (1.0 + composite_std * 0.05)
    final_composite = mean_composite * consistency_factor * stability_factor

    row = asdict(p)
    row.pop("SEARCH_SPACE", None)
    row.update({
        "_n_trades":          total_trades,
        "_win_rate":          cross_avg("win_rate"),
        "_total_return":      cross_avg("total_return"),
        "_profit_factor":     cross_avg("profit_factor"),
        "_max_drawdown":      cross_avg("max_drawdown"),
        "_sharpe":            cross_avg("sharpe"),
        "_avg_duration":      cross_avg("avg_duration"),
        "_composite":         final_composite,
        "_mean_composite":    mean_composite,
        "_consistency":       consistency * 100.0,
        "_composite_std":     composite_std,
        "_n_windows":         n_win,
        "_n_windows_active":  len(active),
        "_final_capital":     initial_capital * (1 + cross_avg("total_return") / 100),
        "_n_symbols_traded":  len([r for r in win_results if r.n_trades > 0]),
    })
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class Optimizer:

    def __init__(self,
                 symbols:         List[str],
                 interval:        str,
                 days:            int,
                 initial_capital: float,
                 min_trades:      int,
                 n_jobs:          int,
                 top_n:           int,
                 output_csv:      str):
        self.symbols         = symbols
        self.interval        = interval
        self.days            = days
        self.initial_capital = initial_capital
        self.min_trades      = min_trades
        self.n_jobs          = n_jobs
        self.top_n           = top_n
        self.output_csv      = output_csv
        self.data_map:        Dict[str, pd.DataFrame] = {}

    def load_data(self, force_download: bool = False):
        print(f"\n{'─'*60}")
        print(f"  Scaricamento dati [{self.interval}] | {self.days} giorni")
        print(f"{'─'*60}")
        for sym in self.symbols:
            df = fetch_klines(sym, self.interval, self.days,
                              force_download=force_download)
            if not df.empty:
                self.data_map[sym] = df
        loaded = list(self.data_map.keys())
        print(f"  Simboli caricati: {len(loaded)}/{len(self.symbols)}: {loaded}")

    def _build_args(self, param_list: List[ParamSet],
                    windows: List[TimeWindow]) -> List[Tuple]:
        return [(p, self.data_map, windows, self.initial_capital, self.min_trades)
                for p in param_list]

    def run(self, param_list: List[ParamSet],
            windows: List[TimeWindow]) -> List[dict]:
        total = len(param_list)
        print(f"\n{'─'*60}")
        print(f"  Avvio ottimizzazione: {total} combinazioni | "
              f"{len(windows)} finestre | {self.n_jobs} job paralleli")
        print(f"{'─'*60}")

        args    = self._build_args(param_list, windows)
        results = []
        t0      = time.time()

        if self.n_jobs <= 1:
            for i, a in enumerate(args, 1):
                r = run_backtest_trial(a)
                if r is not None:
                    results.append(r)
                if i % max(1, total // 20) == 0:
                    elapsed = time.time() - t0
                    eta     = elapsed / i * (total - i)
                    print(f"  [{i:>5}/{total}] trovate {len(results)} config valide "
                          f"| ETA {eta:.0f}s    ", end="\r")
        else:
            chunksize = max(1, total // (self.n_jobs * 4))
            with Pool(processes=self.n_jobs) as pool:
                for i, r in enumerate(
                        pool.imap_unordered(run_backtest_trial, args,
                                            chunksize=chunksize), 1):
                    if r is not None:
                        results.append(r)
                    if i % max(1, total // 20) == 0:
                        elapsed = time.time() - t0
                        eta     = elapsed / i * (total - i)
                        print(f"  [{i:>5}/{total}] trovate {len(results)} config valide "
                              f"| ETA {eta:.0f}s    ", end="\r")

        elapsed = time.time() - t0
        print(f"\n\n  ✓ Completato in {elapsed:.1f}s | "
              f"{len(results)}/{total} config valide "
              f"(min_trades={self.min_trades})")

        results.sort(key=lambda x: x["_composite"], reverse=True)
        return results[:self.top_n * 3]

    def save_results(self, results: List[dict]):
        if not results:
            return
        with open(self.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\n  💾 Risultati CSV salvati in: {self.output_csv}")

    def print_leaderboard(self, results: List[dict],
                          windows: List[TimeWindow]):
        if not results:
            print("\n  ⚠️  Nessuna configurazione valida trovata.")
            return

        top = results[:self.top_n]
        W   = 122

        print(f"\n{'═'*W}")
        print(f"{'  🏆  LEADERBOARD TOP ' + str(self.top_n) + '  (validazione su ' + str(len(windows)) + ' finestre)':^{W}}")
        print(f"{'═'*W}")

        cols = [
            ("Rk",    3), ("EMAf", 5), ("EMAs", 5), ("EMAt", 5),
            ("ADXm",  5), ("RSIob",6), ("SLx",  5), ("TPx",  5),
            ("Thr",   5), ("POS%", 5),
            ("N#",    5), ("WR%",  6), ("Ret%", 7),
            ("PF",    5), ("DD%",  6), ("Sharpe",7),
            ("Cons%", 7), ("Std",  6), ("Score", 7),
        ]
        header = "".join(f"{c[0]:>{c[1]}}" for c in cols)
        print(f"  {header}")
        print(f"  {'─'*(W-2)}")

        for rank, row in enumerate(top, 1):
            line = (
                f"  {rank:>3}"
                f"  {row['ema_fast']:>3}"
                f"  {row['ema_slow']:>3}"
                f"  {row['ema_trend']:>3}"
                f"  {row['adx_min']:>3.0f}"
                f"  {row['rsi_ob']:>4.0f}"
                f"  {row['atr_stop_mult']:>3.1f}"
                f"  {row['atr_tp_mult']:>3.1f}"
                f"  {row['signal_threshold']:>3.0f}"
                f"  {row['position_pct']*100:>4.0f}"
                f"  {row['_n_trades']:>4}"
                f"  {row['_win_rate']:>5.1f}"
                f"  {row['_total_return']:>+6.1f}"
                f"  {row['_profit_factor']:>4.2f}"
                f"  {row['_max_drawdown']:>5.1f}"
                f"  {row['_sharpe']:>+6.2f}"
                f"  {row['_consistency']:>6.1f}"
                f"  {row['_composite_std']:>5.2f}"
                f"  {row['_composite']:>+6.2f}"
            )
            if rank <= 3:
                line = "🥇" + line[1:] if rank == 1 else \
                       "🥈" + line[1:] if rank == 2 else "🥉" + line[1:]
            print(line)

        print(f"  {'─'*(W-2)}")

        # ── Dettaglio configurazione #1 ───────────────────────────────────────
        best = top[0]
        n_win_active = int(best.get("_n_windows_active", len(windows)))
        n_win_total  = int(best.get("_n_windows", len(windows)))

        print(f"\n  ★ CONFIGURAZIONE OTTIMALE (rank #1):")
        print(f"  {'─'*55}")
        
        # ─── INCLUSIONE DELL'INTERVAL E CREAZIONE FILE UNIVOCO ──────────────
        best_clean = {k: v for k, v in best.items() if not k.startswith("_")}
        
        # Aggiungo esplicitamente l'interval dentro il JSON
        best_clean["INTERVAL"] = self.interval
        
        # Creiamo un codice univoco hash basato sui parametri esatti (stringa deterministica)
        param_str = json.dumps(best_clean, sort_keys=True)
        config_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()[:6]
        
        # Costruisco un nome leggibile per il file json
        file_name = f"config_{self.interval}_{best_clean['ema_fast']}-{best_clean['ema_slow']}_{config_hash}.json"
        out_dir = "optimizer_outputs"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, file_name)

        for k, v in best_clean.items():
            print(f"    {k:<25} = {v}")

        print(f"\n  Performance aggregata su {n_win_total} finestre ")
        print(f"    Trade totali   : {best['_n_trades']}  (finestre attive: {n_win_active}/{n_win_total})")
        print(f"    Win rate medio : {best['_win_rate']:.1f}%")
        print(f"    Rendimento medio: {best['_total_return']:+.2f}%")
        print(f"    Score composito: {best['_composite']:+.2f}")
        print(f"  {'─'*55}")

        with open(out_path, "w") as f:
            json.dump(best_clean, f, indent=2)
            
        print(f"\n  ✅ Configurazione univoca salvata in: {out_path}")
        print(f"     Per usarla, copia questo file nella cartella principale e rinominalo in 'best_config.json'")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC",
    "XRPUSDC", "ADAUSDC", "DOGEUSDC", "AVAXUSDC",
]


def main():
    parser = argparse.ArgumentParser(
        description="Ottimizzatore parametri multi-window (ora con salvataggi univoci)"
    )

    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--random", type=int, metavar="N",
                          help="Ricerca casuale: N combinazioni da campionare")
    mode_grp.add_argument("--grid",   action="store_true",
                          help="Grid search: tutte le combinazioni")

    parser.add_argument("--symbols",   default=",".join(DEFAULT_SYMBOLS),
                        help="Simboli separati da virgola (default: tutti)")
    parser.add_argument("--interval",  default="1d",
                        help="Intervallo candele: 15m,1h,4h,1d (default: 1d)")
    parser.add_argument("--days",      type=int,   default=730,
                        help="Giorni di storico da scaricare (default: 730)")
    parser.add_argument("--capital",   type=float, default=100.0,
                        help="Capitale iniziale per il backtest (default: 100)")
    parser.add_argument("--min-trades",type=int,   default=10,
                        help="Numero minimo di trade totali")
    parser.add_argument("--jobs",      type=int,
                        default=max(1, cpu_count() - 1),
                        help="Job paralleli")
    parser.add_argument("--top",       type=int,   default=10,
                        help="Numero di configurazioni top da mostrare (default: 10)")
    parser.add_argument("--out",       default="optimizer_results.csv",
                        help="File CSV di output (default: optimizer_results.csv)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Seed casuale per riproducibilità (default: 42)")
    parser.add_argument("--force-download", action="store_true",
                        help="Forza ri-download dei dati (ignora cache)")
    parser.add_argument("--windows",   type=int,   default=5,
                        help="Numero di finestre temporali casuali")
    parser.add_argument("--win-min",   type=int,   default=90,
                        help="Durata minima di ogni finestra in giorni (default: 90)")
    parser.add_argument("--win-max",   type=int,   default=None,
                        help="Durata massima di ogni finestra in giorni ")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    win_max = args.win_max if args.win_max else args.days // 2
    win_max = min(win_max, args.days - 1)

    opt = Optimizer(
        symbols         = symbols,
        interval        = args.interval,
        days            = args.days,
        initial_capital = args.capital,
        min_trades      = args.min_trades,
        n_jobs          = args.jobs,
        top_n           = args.top,
        output_csv      = args.out,
    )

    opt.load_data(force_download=args.force_download)
    if not opt.data_map:
        sys.exit(1)

    windows = generate_windows(
        data_map  = opt.data_map,
        n_windows = args.windows,
        min_days  = args.win_min,
        max_days  = win_max,
        interval  = args.interval,
        seed      = args.seed,
    )

    print_windows(windows, opt.data_map, args.interval)

    base = ParamSet()
    if args.random:
        param_list = [random_param_set(base) for _ in range(args.random)]
        param_list.append(base)
    else:
        param_list = grid_param_sets(base)

    results = opt.run(param_list, windows)

    opt.save_results(results)
    opt.print_leaderboard(results, windows)


if __name__ == "__main__":
    main()
