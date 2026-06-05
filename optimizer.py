"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CRYPTO STRATEGY OPTIMIZER v1.0                                     ║
║                                                                              ║
║  Modalità di ricerca:                                                        ║
║    --random N    → campiona N combinazioni casuali (default: 300)            ║
║    --grid        → enumera tutte le combo (attenzione: può essere lento)     ║
║                                                                              ║
║  Metriche di ranking (punteggio composito):                                  ║
║    Sharpe ratio  40% · Total return  30% · Win rate  20% · PF  10%          ║
║                                                                              ║
║  Uso:                                                                        ║
║    python optimizer.py --random 500 --days 730 --symbols BTCUSDC,ETHUSDC    ║
║    python optimizer.py --grid   --days 365 --interval 4h                    ║
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
    # SPAZIO DI RICERCA: lista dei valori da testare per ogni parametro.
    # Aggiungi/rimuovi valori per allargare o stringere lo spazio.
    # ──────────────────────────────────────────────────────────────────────────
    SEARCH_SPACE: Dict[str, List[Any]] = None

    def __post_init__(self):
        if self.SEARCH_SPACE is None:
            self.SEARCH_SPACE = {
                # EMA – include valori "brevi" per generare più segnali
                "ema_fast":  [8, 12, 13, 21, 34],
                "ema_slow":  [21, 34, 50, 55, 89],
                "ema_trend": [0, 50, 100, 150, 200],   # 0 = no filtro trend

                # ADX – valori bassi (o 0) generano più trade
                "adx_period": [10, 14, 20],
                "adx_min":    [0, 15, 18, 22, 25],

                # RSI
                "rsi_period": [10, 14, 21],
                "rsi_ob":     [60, 65, 70, 75, 80],
                "rsi_os":     [20, 25, 30, 35],

                # ATR stop/tp
                "atr_period":    [10, 14, 21],
                "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
                "atr_tp_mult":   [2.0, 3.0, 4.0, 5.0],

                # Bollinger
                "bb_period": [15, 20, 25],
                "bb_std":    [1.5, 2.0, 2.5],

                # Volume
                "vol_mult": [1.2, 1.5, 2.0, 2.5, 3.0],

                # Soglia score: abbassarla genera più segnali
                "signal_threshold": [30, 40, 50, 55, 60, 65, 70],

                # Position sizing
                "position_pct": [0.10, 0.15, 0.20, 0.25],

                # Commissioni (fisso, normalmente non ottimizzare)
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
    # Sanity check: ema_fast < ema_slow
    if kwargs["ema_fast"] >= kwargs["ema_slow"]:
        kwargs["ema_fast"] = max(4, kwargs["ema_slow"] - 5)
    # atr_tp > atr_stop (R:R positivo)
    if kwargs["atr_tp_mult"] <= kwargs["atr_stop_mult"]:
        kwargs["atr_tp_mult"] = kwargs["atr_stop_mult"] + 1.0
    return ParamSet(**kwargs)


def grid_param_sets(base: ParamSet) -> List[ParamSet]:
    """Genera tutte le combinazioni possibili (attenzione alle dimensioni!)."""
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
#  DATA CACHE
# ══════════════════════════════════════════════════════════════════════════════

CACHE_DIR = ".ohlcv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(symbol: str, interval: str, days: int) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}_{days}d.parquet")


def fetch_klines(symbol: str, interval: str, days: int,
                 force_download: bool = False) -> pd.DataFrame:
    """
    Scarica le candele OHLCV da Binance e le mette in cache (.parquet).
    Se il file esiste ed è recente (<1h), riusa la cache.
    """
    cache_file = _cache_path(symbol, interval, days)
    if not force_download and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 3600:   # cache valida per 1 ora
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
        time.sleep(0.1)   # evita rate limit

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
    """Ritorna (ADX, +DI, -DI)."""
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
    """
    Calcolo vettoriale di tutti gli indicatori e dei segnali BUY/SELL.

    PUNTEGGIO BUY (max 100):
      25 pt — EMA trend:   close > EMA(ema_trend)  [opzionale se ema_trend==0]
      20 pt — EMA cross:   EMA(fast) > EMA(slow)
      15 pt — ADX:         ADX > adx_min e +DI > -DI [opzionale se adx_min==0]
      15 pt — RSI:         RSI < rsi_ob (bonus pieno se RSI < rsi_os)
      10 pt — Bollinger:   close < BB_middle
      10 pt — Volume:      volume > vol_mult × SMA(volume, 20)
       5 pt — Neutro OB:   fisso 0.5 (order book non disponibile in backtest)

    BUY  → score >= signal_threshold
    SELL → EMA(fast) < EMA(slow)  [conferma con barra precedente]
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # EMAs
    ema_f  = _ema(close, p.ema_fast)
    ema_s  = _ema(close, p.ema_slow)

    # EMA trend filter
    if p.ema_trend > 0:
        ema_t     = _ema(close, p.ema_trend)
        trend_ok  = (close > ema_t).astype(float)
    else:
        trend_ok  = pd.Series(1.0, index=df.index)

    # ADX
    if p.adx_min > 0:
        adx_val, di_plus, di_minus = _adx(df, p.adx_period)
        adx_ok = ((adx_val > p.adx_min) & (di_plus > di_minus)).astype(float)
    else:
        adx_ok = pd.Series(1.0, index=df.index)

    # RSI (punteggio parziale proporzionale alla distanza dall'OB)
    rsi = _rsi(close, p.rsi_period)
    rsi_score = np.where(
        rsi < p.rsi_os, 15.0,                                    # bonus massimo
        np.where(rsi < p.rsi_ob,
                 15.0 * (p.rsi_ob - rsi) / (p.rsi_ob - p.rsi_os + 1e-9),
                 0.0))                                            # OB: zero punti

    # ATR (per SL/TP)
    atr = _atr(df, p.atr_period)

    # Bollinger Bands
    bb_mid = close.rolling(p.bb_period).mean()
    bb_ok  = (close < bb_mid).astype(float)

    # Volume spike
    vol_ma = volume.rolling(20).mean()
    vol_ok = (volume > vol_ma * p.vol_mult).astype(float)

    # Punteggio composito (vettorializzato)
    score = (
        trend_ok * 25.0 +
        (ema_f > ema_s).astype(float) * 20.0 +
        adx_ok  * 15.0 +
        rsi_score +
        bb_ok   * 10.0 +
        vol_ok  * 10.0 +
        2.5                          # OB neutro (fisso al 50%)
    )

    # Segnali
    buy_sig  = (score >= p.signal_threshold)
    # Sell: crossover ribassista (fast attraversa slow dall'alto)
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
    """Metriche di un singolo backtest."""
    symbol:        str
    n_trades:      int
    win_rate:      float
    total_return:  float    # %
    profit_factor: float
    max_drawdown:  float    # %
    sharpe:        float
    avg_duration:  float    # in candele
    final_capital: float
    composite:     float    # punteggio per il ranking


def simulate(sig: pd.DataFrame, p: ParamSet,
             initial_capital: float = 100.0,
             symbol: str = "?") -> BacktestResult:
    """
    Simula le trade con gestione intra-candela di SL e TP.

    Per ogni candela IN posizione:
      - Se low  <= SL → uscita al prezzo SL (ipotesi peggiore intra-barra)
      - Se high >= TP → uscita al prezzo TP
      - Se segnale sell → uscita al close
    Il check SL ha priorità sul TP (conservativo).
    """
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

    n = len(sig)
    # Periodo di warm-up: salta i primi max(ema_trend, ema_slow)+5 indici
    warmup = max(p.ema_trend if p.ema_trend > 0 else 0, p.ema_slow) + 5

    for i in range(warmup, n):
        price = arr_close[i]

        # ── IN POSIZIONE ─────────────────────────────────────────────────
        if position is not None:
            sl    = position["sl"]
            tp    = position["tp"]
            entry = position["entry"]
            qty   = position["qty"]

            exit_price = None
            if arr_low[i] <= sl:
                exit_price = sl           # stop loss colpito intra-barra
            elif arr_high[i] >= tp:
                exit_price = tp           # take profit colpito intra-barra
            elif arr_sell[i]:
                exit_price = price        # segnale di vendita

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

        # ── FUORI POSIZIONE ───────────────────────────────────────────────
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

    # ── Chiudi posizione aperta alla fine del periodo ─────────────────────
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

    # ── Calcola metriche ──────────────────────────────────────────────────
    n_trades = len(trades)
    if n_trades == 0:
        return BacktestResult(symbol, 0, 0, 0, 0, 0, 0, 0, capital, -999)

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls > 0

    gross_profit = pnls[wins].sum()
    gross_loss   = abs(pnls[~wins].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 9.99

    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate     = wins.sum() / n_trades

    # Max Drawdown
    eq   = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / (peak + 1e-9)
    max_dd = float(dd.max() * 100)

    # Sharpe annualizzato
    eq_ret = np.diff(eq) / (eq[:-1] + 1e-9)
    if len(eq_ret) > 1 and eq_ret.std() > 1e-9:
        # Fattore di annualizzazione in base all'intervallo
        sharpe = float(eq_ret.mean() / eq_ret.std() * math.sqrt(252))
    else:
        sharpe = 0.0

    avg_dur = float(np.mean([t["duration"] for t in trades]))

    # Punteggio composito (penalizza meno di 10 trade)
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


def run_backtest_trial(args: Tuple) -> Optional[dict]:
    """
    Funzione libera (compatibile con multiprocessing.Pool).
    Riceve (ParamSet, {symbol: DataFrame}, initial_capital) e ritorna
    un dizionario con le metriche aggregate su tutti i simboli.
    """
    p, data_map, initial_capital, min_trades = args
    results: List[BacktestResult] = []

    for symbol, df in data_map.items():
        if df.empty:
            continue
        try:
            sig = compute_signals(df, p)
            r   = simulate(sig, p, initial_capital, symbol)
            results.append(r)
        except Exception:
            continue

    if not results:
        return None

    # Aggrega su tutti i simboli (media pesata per numero di trade)
    total_trades  = sum(r.n_trades      for r in results)
    if total_trades < min_trades:
        return None

    def wavg(attr, fallback=0.0):
        weights = [max(r.n_trades, 1) for r in results]
        vals    = [getattr(r, attr) for r in results]
        return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

    row = asdict(p)
    row.pop("SEARCH_SPACE", None)
    row.update({
        "_n_trades":      total_trades,
        "_win_rate":      wavg("win_rate"),
        "_total_return":  wavg("total_return"),
        "_profit_factor": wavg("profit_factor"),
        "_max_drawdown":  wavg("max_drawdown"),
        "_sharpe":        wavg("sharpe"),
        "_avg_duration":  wavg("avg_duration"),
        "_composite":     wavg("composite"),
        "_final_capital": sum(r.final_capital for r in results),
        "_n_symbols_traded": len([r for r in results if r.n_trades > 0]),
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

    def load_data(self):
        print(f"\n{'─'*60}")
        print(f"  Scaricamento dati [{self.interval}] | {self.days} giorni")
        print(f"{'─'*60}")
        for sym in self.symbols:
            df = fetch_klines(sym, self.interval, self.days)
            if not df.empty:
                self.data_map[sym] = df
        loaded = list(self.data_map.keys())
        print(f"  Simboli caricati: {len(loaded)}/{len(self.symbols)}: {loaded}")

    def _build_args(self, param_list: List[ParamSet]):
        return [(p, self.data_map, self.initial_capital, self.min_trades)
                for p in param_list]

    def run(self, param_list: List[ParamSet]) -> List[dict]:
        total = len(param_list)
        print(f"\n{'─'*60}")
        print(f"  Avvio ottimizzazione: {total} combinazioni | "
              f"{self.n_jobs} job paralleli")
        print(f"{'─'*60}")

        args    = self._build_args(param_list)
        results = []
        t0      = time.time()

        if self.n_jobs <= 1:
            # Esecuzione sequenziale (utile per debug o sistemi con poca RAM)
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

        # Ordina per punteggio composito
        results.sort(key=lambda x: x["_composite"], reverse=True)
        return results[:self.top_n * 3]   # buffer per ulteriori filtri

    def save_results(self, results: List[dict]):
        if not results:
            return
        with open(self.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\n  💾 Risultati salvati in: {self.output_csv}")

    def print_leaderboard(self, results: List[dict]):
        if not results:
            print("\n  ⚠️  Nessuna configurazione valida trovata.")
            print("     Prova a ridurre --min-trades o ad allargare lo spazio di ricerca.")
            return

        top = results[:self.top_n]
        W   = 110

        print(f"\n{'═'*W}")
        print(f"{'  🏆  LEADERBOARD TOP ' + str(self.top_n):^{W}}")
        print(f"{'═'*W}")

        # Header tabella
        cols = [
            ("Rk", 3), ("EMAf", 5), ("EMAs", 5), ("EMAt", 5),
            ("ADXm", 5), ("RSIob", 6), ("SLx", 5), ("TPx", 5),
            ("Thr", 5), ("POS%", 5),
            ("N#", 5), ("WR%", 6), ("Ret%", 7),
            ("PF", 5), ("DD%", 6), ("Sharpe", 7), ("Score", 7),
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
                f"  {row['_composite']:>+6.2f}"
            )
            # Colorazione testuale (verde per le prime 3 posizioni)
            if rank <= 3:
                line = "🥇" + line[1:] if rank == 1 else \
                       "🥈" + line[1:] if rank == 2 else "🥉" + line[1:]
            print(line)

        print(f"  {'─'*(W-2)}")
        print(f"  Colonne: EMAf/s/t=periodi EMA fast/slow/trend | ADXm=ADX min | "
              f"RSIob=RSI overbought")
        print(f"           SLx/TPx=moltiplicatori ATR | Thr=soglia score | "
              f"POS%=% capitale/trade")
        print(f"           N#=N.trade | WR%=winrate | Ret%=rendimento | "
              f"PF=profit factor | DD%=max drawdown")
        print(f"{'═'*W}")

        # Dettaglio configurazione #1
        best = top[0]
        print(f"\n  ★ CONFIGURAZIONE OTTIMALE (rank #1):")
        print(f"  {'─'*50}")
        param_keys = [f.name for f in fields(ParamSet())
                      if f.name != "SEARCH_SPACE"]
        for k in param_keys:
            if k in best:
                print(f"    {k:<25} = {best[k]}")
        print(f"\n  Performance su [{self.interval}] {self.days}d "
              f"capitale {self.initial_capital:.0f} USDC:")
        print(f"    Trade totali  : {best['_n_trades']}")
        print(f"    Win rate      : {best['_win_rate']:.1f}%")
        print(f"    Rendimento    : {best['_total_return']:+.2f}%")
        print(f"    Profit Factor : {best['_profit_factor']:.2f}")
        print(f"    Max Drawdown  : {best['_max_drawdown']:.2f}%")
        print(f"    Sharpe ratio  : {best['_sharpe']:.2f}")
        print(f"    Capitale finale: {best['_final_capital']:.2f} USDC")
        print(f"  {'─'*50}")

        # Snippet Python per copiare direttamente nel bot
        print(f"\n  📋 SNIPPET — copia in Config() del tuo bot:")
        print(f"  {'─'*50}")
        print(f"  cfg = Config(")
        snippet_keys = [
            "ema_fast", "ema_slow", "ema_trend",
            "adx_period", "adx_min",
            "rsi_period", "rsi_ob", "rsi_os",
            "atr_period", "atr_stop_mult", "atr_tp_mult",
            "bb_period", "bb_std",
            "vol_mult", "signal_threshold", "position_pct",
        ]
        for k in snippet_keys:
            if k in best:
                v = best[k]
                print(f"      {k.upper():<25} = {v},")
        print(f"  )")
        print(f"  {'─'*50}")

        # Salva anche la best config in JSON
        best_clean = {k: v for k, v in best.items() if not k.startswith("_")}
        with open("best_config.json", "w") as f:
            json.dump(best_clean, f, indent=2)
        print(f"\n  💾 Best config salvata in: best_config.json")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC",
    "XRPUSDC", "ADAUSDC", "DOGEUSDC", "AVAXUSDC",
]


def main():
    parser = argparse.ArgumentParser(
        description="Ottimizzatore parametri per il bot di trading crypto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python optimizer.py --random 500 --days 730
  python optimizer.py --random 200 --days 365 --interval 4h --symbols BTCUSDC,ETHUSDC
  python optimizer.py --grid   --days 365 --interval 1d --jobs 2
  python optimizer.py --random 1000 --min-trades 20 --top 20
        """)

    # Modalità ricerca
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--random", type=int, metavar="N",
                          help="Ricerca casuale: N combinazioni da campionare")
    mode_grp.add_argument("--grid",   action="store_true",
                          help="Grid search: tutte le combinazioni (può essere lento!)")

    # Configurazione
    parser.add_argument("--symbols",   default=",".join(DEFAULT_SYMBOLS),
                        help=f"Simboli separati da virgola (default: tutti)")
    parser.add_argument("--interval",  default="1d",
                        help="Intervallo candele: 15m,1h,4h,1d (default: 1d)")
    parser.add_argument("--days",      type=int,   default=730,
                        help="Giorni di storico da scaricare (default: 730)")
    parser.add_argument("--capital",   type=float, default=100.0,
                        help="Capitale iniziale per il backtest (default: 100)")
    parser.add_argument("--min-trades",type=int,   default=10,
                        help="Numero minimo di trade per considerare la config (default: 10)")
    parser.add_argument("--jobs",      type=int,
                        default=max(1, cpu_count() - 1),
                        help=f"Job paralleli (default: {max(1,cpu_count()-1)}, "
                             f"Raspberry Pi: usa 1 o 2)")
    parser.add_argument("--top",       type=int,   default=10,
                        help="Numero di configurazioni top da mostrare (default: 10)")
    parser.add_argument("--out",       default="optimizer_results.csv",
                        help="File CSV di output (default: optimizer_results.csv)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Seed casuale per riproducibilità (default: 42)")
    parser.add_argument("--force-download", action="store_true",
                        help="Forza ri-download dei dati (ignora cache)")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            CRYPTO STRATEGY OPTIMIZER v1.0                   ║
╠══════════════════════════════════════════════════════════════╣
║  Simboli   : {', '.join(symbols):<43} ║
║  Intervallo: {args.interval:<47} ║
║  Periodo   : {args.days} giorni{'':<39} ║
║  Capitale  : {args.capital:.0f} USDC{'':<42} ║
║  Min trade : {args.min_trades:<47} ║
║  Jobs      : {args.jobs:<47} ║
╚══════════════════════════════════════════════════════════════╝""")

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

    # 1. Scarica dati
    opt.load_data()
    if not opt.data_map:
        print("  ERRORE: nessun dato scaricato. Controlla la connessione o i simboli.")
        sys.exit(1)

    # 2. Genera lista di ParamSet
    base = ParamSet()
    if args.random:
        n = args.random
        print(f"\n  Generazione {n} combinazioni casuali (seed={args.seed})...")
        param_list = [random_param_set(base) for _ in range(n)]
        # Aggiungi sempre il ParamSet di default come punto di riferimento
        param_list.append(base)
    else:
        print(f"\n  Generazione grid search (spazio completo)...")
        param_list = grid_param_sets(base)
        grid_size  = len(param_list)
        print(f"  Grid size: {grid_size} combinazioni")
        if grid_size > 10_000:
            print(f"  ⚠️  Grid molto grande! Usa --random per la ricerca casuale.")
            confirm = input("  Procedere? [s/N]: ")
            if confirm.lower() != "s":
                sys.exit(0)

    # 3. Esegui ottimizzazione
    results = opt.run(param_list)

    # 4. Salva e stampa risultati
    opt.save_results(results)
    opt.print_leaderboard(results)


if __name__ == "__main__":
    main()