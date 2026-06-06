"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CRYPTO STRATEGY OPTIMIZER v2.0                                             ║
║  Multi-interval · Walk-Forward · Latin Hypercube · RPi4-ready               ║
║                                                                             ║
║  Pipeline per ogni interval:                                                ║
║    Fase 1 — Ottimizzazione IS (primo 80% dati, parallela)                  ║
║    Fase 2 — Walk-Forward su IS (top 150, 3 finestre espandibili)            ║
║    Fase 3 — Out-Of-Sample test (top 30 da WF, ultimo 20% dati)             ║
║    Score finale = IS×40% + WF×35% + OOS×25%                               ║
║                                                                             ║
║  Output: optimizer_{interval}.csv · best_config_{interval}.json            ║
║          best_configs_all.json (riepilogo tutti gli interval)               ║
║                                                                             ║
║  Uso:                                                                       ║
║    python optimizer.py --random 500                                         ║
║    python optimizer.py --random 1000 --intervals 1h,4h,1d --days 730       ║
║    python optimizer.py --random 2000 --wf-splits 4 --symbols BTCUSDC,...   ║
║    python optimizer.py --grid   --intervals 1d --days 365                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

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
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  COSTANTI GLOBALI
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC",
    "XRPUSDC", "ADAUSDC", "DOGEUSDC", "AVAXUSDC",
]

# Percentuale di dati riservata come holdout OOS (mai vista durante ottimizzazione)
OOS_FRACTION = 0.20

# Quanti candidati portare alle fasi WF e OOS
WF_TOP_K  = 150
OOS_TOP_K = 30

CACHE_DIR = ".ohlcv_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE ESTESO
#  Ogni lista contiene i valori discreti da esplorare per quel parametro.
#  Aggiungi/rimuovi valori per allargare/stringere lo spazio.
# ══════════════════════════════════════════════════════════════════════════════

SEARCH_SPACE: Dict[str, List[Any]] = {
    # EMA — breve gamma per segnali frequenti, lunga per trend stabili
    "ema_fast":  [5, 8, 10, 12, 13, 16, 21, 26, 34],
    "ema_slow":  [18, 21, 26, 34, 40, 50, 55, 64, 89],
    "ema_trend": [0, 50, 75, 100, 150, 200],   # 0 = disabilita filtro trend

    # ADX — valori bassi generano più trade ma più rumorosi
    "adx_period": [7, 10, 14, 20],
    "adx_min":    [0, 12, 15, 18, 20, 22, 25, 28],   # 0 = disabilita

    # RSI
    "rsi_period": [7, 10, 14, 21],
    "rsi_ob":     [55, 60, 65, 70, 75, 80],
    "rsi_os":     [20, 25, 30, 35, 40],

    # ATR stop/TP — rapporto TP/SL determina il R:R
    "atr_period":    [7, 10, 14, 21],
    "atr_stop_mult": [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
    "atr_tp_mult":   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],

    # Bollinger Bands
    "bb_period": [10, 15, 20, 25, 30],
    "bb_std":    [1.5, 1.8, 2.0, 2.2, 2.5, 3.0],

    # Volume spike — moltiplicatore sulla SMA(20) del volume
    "vol_mult": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],

    # Soglia score: abbassarla → più segnali ma più falsi positivi
    "signal_threshold": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],

    # Sizing: % del capitale per singolo trade
    "position_pct": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],

    # Commissioni — fisso, normalmente non ottimizzare
    "commission": [0.001],
}


# ══════════════════════════════════════════════════════════════════════════════
#  PARAM SET + CAMPIONAMENTO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamSet:
    ema_fast:         int   = 21
    ema_slow:         int   = 55
    ema_trend:        int   = 200
    adx_period:       int   = 14
    adx_min:          float = 22.0
    rsi_period:       int   = 14
    rsi_ob:           float = 70.0
    rsi_os:           float = 30.0
    atr_period:       int   = 14
    atr_stop_mult:    float = 2.0
    atr_tp_mult:      float = 3.0
    bb_period:        int   = 20
    bb_std:           float = 2.0
    vol_mult:         float = 1.5
    signal_threshold: float = 55.0
    position_pct:     float = 0.15
    commission:       float = 0.001


def _apply_constraints(kw: dict) -> dict:
    """Forza i vincoli logici: ema_fast < ema_slow, atr_tp > atr_stop."""
    if kw["ema_fast"] >= kw["ema_slow"]:
        kw["ema_fast"] = max(4, kw["ema_slow"] - 5)
    if kw["atr_tp_mult"] <= kw["atr_stop_mult"]:
        kw["atr_tp_mult"] = kw["atr_stop_mult"] + 1.0
    return kw


def random_param() -> ParamSet:
    """Campiona un ParamSet casuale uniform."""
    kw = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    return ParamSet(**_apply_constraints(kw))


def lhs_params(n: int) -> List[ParamSet]:
    """
    Latin Hypercube Sampling: ogni valore del search space compare
    approssimativamente n/k volte (k = numero valori del parametro).
    Garantisce copertura più uniforme del pure random.
    """
    cols: Dict[str, list] = {}
    for key, vals in SEARCH_SPACE.items():
        k = len(vals)
        col = [vals[i % k] for i in range(n)]
        random.shuffle(col)
        cols[key] = col
    result = []
    for i in range(n):
        kw = {k: cols[k][i] for k in SEARCH_SPACE}
        result.append(ParamSet(**_apply_constraints(kw)))
    return result


def grid_params() -> List[ParamSet]:
    """Genera tutte le combinazioni valide del search space."""
    keys, vals = list(SEARCH_SPACE.keys()), list(SEARCH_SPACE.values())
    out = []
    for combo in itertools.product(*vals):
        kw = dict(zip(keys, combo))
        if kw["ema_fast"] < kw["ema_slow"] and kw["atr_tp_mult"] > kw["atr_stop_mult"]:
            out.append(ParamSet(**kw))
    return out


def param_from_dict(row: dict) -> ParamSet:
    """Ricostruisce un ParamSet da un dizionario (ignora chiavi con prefisso _)."""
    valid = {k: v for k, v in row.items()
             if not k.startswith("_") and k in ParamSet.__dataclass_fields__}
    return ParamSet(**valid)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CACHE — download da Binance con cache .parquet (1h di validità)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_klines(symbol: str, interval: str, days: int,
                 force: bool = False) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{days}d.parquet")

    if not force and os.path.exists(path):
        if time.time() - os.path.getmtime(path) < 3600:
            return pd.read_parquet(path)

    print(f"  ↓ {symbol} [{interval}] {days}d ...", end="", flush=True)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    url = "https://api.binance.com/api/v3/klines"
    all_rows, current = [], start_dt

    while True:
        try:
            r = requests.get(url, params={
                "symbol": symbol, "interval": interval, "limit": 1000,
                "startTime": int(current.timestamp() * 1000),
                "endTime":   int(end_dt.timestamp() * 1000),
            }, timeout=15)
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            print(f" ERR: {e}")
            return pd.DataFrame()

        if not rows:
            break
        all_rows.extend(rows)
        last_ts = pd.Timestamp(rows[-1][0], unit="ms")
        current = last_ts + timedelta(milliseconds=1)
        if len(rows) < 1000 or last_ts >= end_dt:
            break
        time.sleep(0.1)

    if not all_rows:
        print(" [no data]")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df.to_parquet(path, index=False)
    print(f" {len(df)} candles OK")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORI (vettoriali)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    h, l = df["high"], df["low"]
    atr  = _atr(df, n)
    up   = h.diff().clip(lower=0)
    dn   = (-l.diff()).clip(lower=0)
    up[up < dn] = 0
    dn[dn < up] = 0
    a    = 1 / n
    dip  = 100 * up.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    dim  = 100 * dn.ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    dx   = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean(), dip, dim


# ══════════════════════════════════════════════════════════════════════════════
#  CALCOLO SEGNALI
#
#  Punteggio BUY (0-100):
#    25 pt — EMA trend:  close > EMA(ema_trend)   [skip se ema_trend=0]
#    20 pt — EMA cross:  EMA_fast > EMA_slow
#    15 pt — ADX:        ADX > adx_min e +DI > -DI [skip se adx_min=0]
#    15 pt — RSI:        proporzionale distanza da OB (pieno se < OS)
#    10 pt — Bollinger:  close < BB_middle
#    10 pt — Volume:     volume > vol_mult × SMA(20)
#     2.5 pt — bonus fisso (OB neutro in backtest)
#
#  BUY  → score >= signal_threshold
#  SELL → crossover ribassista EMA (fast < slow dopo essere stato >)
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(df: pd.DataFrame, p: ParamSet) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    ef = _ema(c, p.ema_fast)
    es = _ema(c, p.ema_slow)

    if p.ema_trend > 0:
        trend_ok = (c > _ema(c, p.ema_trend)).astype(float)
    else:
        trend_ok = pd.Series(1.0, index=df.index)

    if p.adx_min > 0:
        adx_val, dip, dim = _adx(df, p.adx_period)
        adx_ok = ((adx_val > p.adx_min) & (dip > dim)).astype(float)
    else:
        adx_ok = pd.Series(1.0, index=df.index)

    rsi = _rsi(c, p.rsi_period)
    rsi_score = np.where(
        rsi < p.rsi_os, 15.0,
        np.where(rsi < p.rsi_ob,
                 15.0 * (p.rsi_ob - rsi) / (p.rsi_ob - p.rsi_os + 1e-9),
                 0.0))

    atr    = _atr(df, p.atr_period)
    bb_mid = c.rolling(p.bb_period).mean()
    vol_ma = v.rolling(20).mean()

    score = (
        trend_ok * 25.0 +
        (ef > es).astype(float) * 20.0 +
        adx_ok * 15.0 +
        rsi_score +
        (c < bb_mid).astype(float) * 10.0 +
        (v > vol_ma * p.vol_mult).astype(float) * 10.0 +
        2.5
    )

    return pd.DataFrame({
        "close": c.values, "high": h.values, "low": l.values,
        "atr":   atr.values, "score": score.values,
        "buy":  (score >= p.signal_threshold).values,
        "sell": ((ef < es) & (ef.shift(1) >= es.shift(1))).values,
    }, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    symbol:        str
    n_trades:      int
    win_rate:      float
    total_return:  float   # % rispetto al capitale iniziale
    profit_factor: float
    max_drawdown:  float   # %
    sharpe:        float   # annualizzato
    avg_duration:  float   # candele
    composite:     float   # score ponderato per il ranking


def simulate(sig: pd.DataFrame, p: ParamSet,
             initial_capital: float = 100.0,
             symbol: str = "?") -> BacktestResult:
    """
    Simula la strategia con gestione intra-candela di SL e TP.
    SL ha priorità sul TP (conservativo).
    """
    capital = initial_capital
    pos     = None
    trades: List[dict] = []
    equity  = [capital]

    arr_c  = sig["close"].values
    arr_h  = sig["high"].values
    arr_l  = sig["low"].values
    arr_a  = sig["atr"].values
    arr_b  = sig["buy"].values
    arr_s  = sig["sell"].values
    n      = len(sig)

    warmup = max(p.ema_trend if p.ema_trend > 0 else 0, p.ema_slow) + 5

    for i in range(warmup, n):
        px = arr_c[i]

        if pos is not None:
            ep = None
            if arr_l[i] <= pos["sl"]:    ep = pos["sl"]   # SL colpito
            elif arr_h[i] >= pos["tp"]:  ep = pos["tp"]   # TP colpito
            elif arr_s[i]:               ep = px           # segnale sell

            if ep is not None:
                gross  = (ep - pos["entry"]) * pos["qty"]
                comm   = (ep + pos["entry"]) * pos["qty"] * p.commission
                pnl    = gross - comm
                capital += pnl
                trades.append({"pnl": pnl, "inv": pos["inv"],
                                "dur": i - pos["idx"]})
                pos = None

        if pos is None and arr_b[i] and capital >= 10:
            inv = min(capital * p.position_pct, capital)
            if inv < 5:
                equity.append(capital)
                continue
            atr = arr_a[i]
            qty = (inv * (1 - p.commission)) / px
            pos = {
                "entry": px,
                "sl":    px - p.atr_stop_mult * atr,
                "tp":    px + p.atr_tp_mult  * atr,
                "qty":   qty, "inv": inv, "idx": i,
            }

        equity.append(capital)

    # Chiude posizione aperta a fine periodo al prezzo di close
    if pos is not None:
        pnl = (arr_c[-1] - pos["entry"]) * pos["qty"]
        capital += pnl
        trades.append({"pnl": pnl, "inv": pos["inv"], "dur": n - 1 - pos["idx"]})
        equity[-1] = capital

    if not trades:
        return BacktestResult(symbol, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -999.0)

    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls > 0
    gp   = pnls[wins].sum()
    gl   = abs(pnls[~wins].sum())
    pf   = gp / gl if gl > 0 else 9.99
    wr   = float(wins.sum() / len(trades))
    ret  = (capital - initial_capital) / initial_capital * 100

    eq   = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    mdd  = float(((peak - eq) / (peak + 1e-9)).max() * 100)

    eq_r = np.diff(eq) / (eq[:-1] + 1e-9)
    sh   = (float(eq_r.mean() / eq_r.std() * math.sqrt(252))
            if len(eq_r) > 1 and eq_r.std() > 1e-9 else 0.0)

    # Penalizza configurazioni con pochi trade (< 15)
    trade_bonus = min(1.0, len(trades) / 15)
    comp = (sh * 0.40 + ret * 0.30 + wr * 100 * 0.20 +
            min(pf, 5.0) * 2 * 0.10) * trade_bonus

    return BacktestResult(
        symbol=symbol, n_trades=len(trades),
        win_rate=float(wr * 100), total_return=float(ret),
        profit_factor=float(min(pf, 9.99)), max_drawdown=float(mdd),
        sharpe=float(sh), avg_duration=float(np.mean([t["dur"] for t in trades])),
        composite=float(comp),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TRIAL PARALLELO (funzione libera per multiprocessing.Pool)
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(args: Tuple) -> Optional[dict]:
    """Esegue un backtest su tutti i simboli e aggrega le metriche."""
    p, data_map, min_trades = args
    results: List[BacktestResult] = []

    for sym, df in data_map.items():
        if df.empty:
            continue
        try:
            sig = compute_signals(df, p)
            r   = simulate(sig, p, 100.0, sym)
            results.append(r)
        except Exception:
            continue

    if not results:
        return None

    total = sum(r.n_trades for r in results)
    if total < min_trades:
        return None

    def wavg(attr: str) -> float:
        ws = [max(r.n_trades, 1) for r in results]
        vs = [getattr(r, attr) for r in results]
        return sum(v * w for v, w in zip(vs, ws)) / sum(ws)

    row = asdict(p)
    row.update({
        "_n_trades":      total,
        "_win_rate":      wavg("win_rate"),
        "_total_return":  wavg("total_return"),
        "_profit_factor": wavg("profit_factor"),
        "_max_drawdown":  wavg("max_drawdown"),
        "_sharpe":        wavg("sharpe"),
        "_avg_duration":  wavg("avg_duration"),
        "_composite":     wavg("composite"),
        "_n_symbols":     len([r for r in results if r.n_trades > 0]),
    })
    return row


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD + OOS VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _make_wf_windows(df: pd.DataFrame,
                     n_splits: int) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genera n_splits finestre di test walk-forward ancorate all'interno
    della porzione IS (escluso l'ultimo OOS_FRACTION).
    Pattern espandibile: il training cresce ad ogni split.

    Esempio con n_splits=3 su IS (80% totale):
      Split 0: train=[0:26%], test=[26:53%]
      Split 1: train=[0:53%], test=[53:80%]  ← IS completo
      (la finestra [80:100%] è riservata come OOS finale)
    """
    T      = len(df)
    is_end = int(T * (1 - OOS_FRACTION))
    is_df  = df.iloc[:is_end]
    T_is   = len(is_df)
    step   = T_is // (n_splits + 1)

    windows = []
    for i in range(n_splits):
        tr_end = step * (i + 1)
        te_end = step * (i + 2)
        tr = is_df.iloc[:tr_end]
        te = is_df.iloc[tr_end:te_end]
        if len(tr) > 50 and len(te) > 10:
            windows.append((tr, te))
    return windows


def _wf_score(row: dict, data_map: Dict[str, pd.DataFrame],
              n_splits: int) -> float:
    """
    Calcola lo score walk-forward medio sui test windows IS.
    Usa solo i dati IS (le finestre escludono automaticamente OOS).
    """
    p = param_from_dict(row)
    scores: List[float] = []

    for sym, df in data_map.items():
        if df.empty or len(df) < 60:
            continue
        for _, test_df in _make_wf_windows(df, n_splits):
            try:
                sig = compute_signals(test_df, p)
                r   = simulate(sig, p, 100.0, sym)
                if r.n_trades >= 1:
                    scores.append(r.composite)
            except Exception:
                continue

    return float(np.mean(scores)) if scores else -999.0


def _oos_score(row: dict, data_map: Dict[str, pd.DataFrame]) -> float:
    """
    Testa il ParamSet sull'holdout OOS (ultimo OOS_FRACTION dei dati).
    Questa sezione non viene mai vista durante l'ottimizzazione.
    """
    p = param_from_dict(row)
    scores: List[float] = []

    for sym, df in data_map.items():
        if df.empty:
            continue
        start = int(len(df) * (1 - OOS_FRACTION))
        oos   = df.iloc[start:].reset_index(drop=True)
        if len(oos) < 20:
            continue
        try:
            sig = compute_signals(oos, p)
            r   = simulate(sig, p, 100.0, sym)
            if r.n_trades >= 1:
                scores.append(r.composite)
        except Exception:
            continue

    return float(np.mean(scores)) if scores else -999.0


def _final_score(is_s: float, wf_s: float, oos_s: float) -> float:
    """
    Combina IS, WF e OOS con pesi ponderati.
    Ignora componenti non calcolate (< -100) normalizzando i pesi.
    """
    parts = []
    if is_s  > -100: parts.append((is_s,  0.40))
    if wf_s  > -100: parts.append((wf_s,  0.35))
    if oos_s > -100: parts.append((oos_s, 0.25))
    if not parts:
        return -999.0
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class Optimizer:

    def __init__(self, symbols: List[str], days: int, min_trades: int,
                 n_jobs: int, top_n: int, wf_splits: int, out_prefix: str):
        self.symbols    = symbols
        self.days       = days
        self.min_trades = min_trades
        self.n_jobs     = n_jobs
        self.top_n      = top_n
        self.wf_splits  = wf_splits
        self.prefix     = out_prefix
        # Cache dati: {interval → {symbol → DataFrame}}
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    # ── Dati ─────────────────────────────────────────────────────────────────

    def load_data(self, interval: str, force: bool = False) -> Dict[str, pd.DataFrame]:
        if interval in self._cache and not force:
            return self._cache[interval]
        print(f"\n  Dati [{interval}] — {self.days} giorni:")
        dm: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = fetch_klines(sym, interval, self.days, force)
            if not df.empty:
                dm[sym] = df
        print(f"  Caricati: {len(dm)}/{len(self.symbols)} simboli")
        self._cache[interval] = dm
        return dm

    def _is_map(self, data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Ritaglia ogni serie al solo periodo IS (primo 1-OOS_FRACTION)."""
        return {
            sym: df.iloc[:int(len(df) * (1 - OOS_FRACTION))].reset_index(drop=True)
            for sym, df in data_map.items() if not df.empty
        }

    # ── Fase 1: Ottimizzazione IS parallela ──────────────────────────────────

    def _phase1_optimize(self, param_list: List[ParamSet],
                         is_map: Dict[str, pd.DataFrame]) -> List[dict]:
        total  = len(param_list)
        args   = [(p, is_map, self.min_trades) for p in param_list]
        results: List[dict] = []
        t0     = time.time()

        if self.n_jobs <= 1:
            for i, a in enumerate(args, 1):
                r = run_trial(a)
                if r:
                    results.append(r)
                if i % max(1, total // 10) == 0:
                    self._progress(i, total, len(results), t0)
        else:
            cs = max(1, total // (self.n_jobs * 8))
            with Pool(self.n_jobs) as pool:
                for i, r in enumerate(
                        pool.imap_unordered(run_trial, args, chunksize=cs), 1):
                    if r:
                        results.append(r)
                    if i % max(1, total // 10) == 0:
                        self._progress(i, total, len(results), t0)

        print(f"\n    ✓ Fase 1 completata: {len(results)}/{total} config valide "
              f"({time.time()-t0:.0f}s)")
        results.sort(key=lambda x: x["_composite"], reverse=True)
        return results

    # ── Fase 2: Walk-Forward Validation ──────────────────────────────────────

    def _phase2_wf(self, candidates: List[dict],
                   data_map: Dict[str, pd.DataFrame]) -> List[dict]:
        k = min(WF_TOP_K, len(candidates))
        print(f"\n    Fase 2 — Walk-Forward: top {k} candidati "
              f"({self.wf_splits} split IS)...")
        t0 = time.time()

        for i, row in enumerate(candidates[:k], 1):
            row["_wf_score"] = _wf_score(row, data_map, self.wf_splits)
            if i % max(1, k // 5) == 0:
                print(f"    WF {i}/{k}  ({time.time()-t0:.0f}s)    ", end="\r")

        for row in candidates[k:]:
            row["_wf_score"] = -999.0

        print(f"\n    ✓ Fase 2 completata ({time.time()-t0:.0f}s)")
        return candidates

    # ── Fase 3: Out-Of-Sample Test ────────────────────────────────────────────

    def _phase3_oos(self, candidates: List[dict],
                    data_map: Dict[str, pd.DataFrame]) -> List[dict]:
        # Prende i migliori per WF score
        wf_sorted = sorted(candidates,
                           key=lambda x: x.get("_wf_score", -999), reverse=True)
        k = min(OOS_TOP_K, len(wf_sorted))
        print(f"\n    Fase 3 — OOS test: top {k} (per WF score)...")
        t0 = time.time()

        for row in wf_sorted[:k]:
            row["_oos_score"] = _oos_score(row, data_map)
        for row in wf_sorted[k:]:
            row["_oos_score"] = -999.0

        # Score finale per ogni candidato
        for row in candidates:
            row["_final_score"] = _final_score(
                row.get("_composite", -999),
                row.get("_wf_score",  -999),
                row.get("_oos_score", -999),
            )

        print(f"    ✓ Fase 3 completata ({time.time()-t0:.0f}s)")
        return candidates

    # ── Esecuzione per singolo interval ──────────────────────────────────────

    def run_interval(self, interval: str,
                     param_list: List[ParamSet]) -> List[dict]:
        """
        Esegue l'intera pipeline (IS → WF → OOS) per un singolo interval.
        Ritorna la lista di risultati ordinata per _final_score.
        """
        data_map = self.load_data(interval)
        if not data_map:
            print(f"  Nessun dato disponibile per [{interval}]. Skip.")
            return []

        is_map = self._is_map(data_map)
        n_is   = next(len(v) for v in is_map.values())
        n_oos  = next(int(len(v) * OOS_FRACTION) for v in data_map.values())

        print(f"\n{'─'*68}")
        print(f"  INTERVAL: {interval}  |  {len(param_list)} combinazioni  |  "
              f"{len(is_map)} simboli")
        print(f"  IS: ~{n_is} candles  |  OOS: ~{n_oos} candles  |  "
              f"WF splits: {self.wf_splits}")
        print(f"{'─'*68}")

        results = self._phase1_optimize(param_list, is_map)
        if not results:
            print(f"  ⚠️  Nessuna config valida per [{interval}].")
            return []

        results = self._phase2_wf(results, data_map)
        results = self._phase3_oos(results, data_map)
        results.sort(key=lambda x: x.get("_final_score", -999), reverse=True)
        return results

    # ── Output ────────────────────────────────────────────────────────────────

    def save(self, results: List[dict], interval: str) -> Tuple[str, str]:
        if not results:
            return "", ""

        csv_path  = f"{self.prefix}_{interval}.csv"
        json_path = f"best_config_{interval}.json"

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

        best_clean = {k: v for k, v in results[0].items()
                      if not k.startswith("_")}
        with open(json_path, "w") as f:
            json.dump(best_clean, f, indent=2)

        print(f"\n  💾 {csv_path}  |  💾 {json_path}")
        return csv_path, json_path

    def print_leaderboard(self, results: List[dict], interval: str):
        if not results:
            return
        top = results[:self.top_n]
        W   = 128

        print(f"\n{'═'*W}")
        print(f"{'  🏆  TOP ' + str(self.top_n) + '  —  INTERVAL: ' + interval:^{W}}")
        print(f"{'═'*W}")
        print(
            "  Rk  EMAf EMAs EMAt ADXm RSIob SLx  TPx  Thr Pos%"
            "  N#    WR%   Ret%   PF    DD%  Sharpe"
            "   IS     WF    OOS  Final")
        print(f"  {'─'*(W-2)}")

        medals = {1: "🥇", 2: "🥈", 3: "🥉"}
        for rank, row in enumerate(top, 1):
            pre = medals.get(rank, f"  {rank:2}")
            line = (
                f"{pre}"
                f"  {row['ema_fast']:>3}"
                f"  {row['ema_slow']:>3}"
                f"  {row['ema_trend']:>3}"
                f"  {row['adx_min']:>3.0f}"
                f"  {row['rsi_ob']:>4.0f}"
                f"  {row['atr_stop_mult']:>3.1f}"
                f"  {row['atr_tp_mult']:>3.1f}"
                f"  {row['signal_threshold']:>3.0f}"
                f"  {row['position_pct']*100:>3.0f}%"
                f"  {row['_n_trades']:>4}"
                f"  {row['_win_rate']:>5.1f}"
                f"  {row['_total_return']:>+6.1f}"
                f"  {row['_profit_factor']:>4.2f}"
                f"  {row['_max_drawdown']:>5.1f}"
                f"  {row['_sharpe']:>+6.2f}"
                f"  {row['_composite']:>+5.1f}"
                f"  {row.get('_wf_score', -999):>+5.1f}"
                f"  {row.get('_oos_score', -999):>+5.1f}"
                f"  {row.get('_final_score', -999):>+6.2f}"
            )
            print(line)

        print(f"  {'─'*(W-2)}")

        best = top[0]
        print(f"\n  ★  BEST CONFIG [{interval}]")
        print(f"  {'─'*52}")
        for k in ["ema_fast", "ema_slow", "ema_trend",
                  "adx_period", "adx_min",
                  "rsi_period", "rsi_ob", "rsi_os",
                  "atr_period", "atr_stop_mult", "atr_tp_mult",
                  "bb_period", "bb_std", "vol_mult",
                  "signal_threshold", "position_pct"]:
            if k in best:
                print(f"    {k:<25} = {best[k]}")

        print(f"\n  Metriche [{interval}]:")
        print(f"    Trade : {best['_n_trades']}  |  WR: {best['_win_rate']:.1f}%  "
              f"|  Return: {best['_total_return']:+.2f}%")
        print(f"    PF    : {best['_profit_factor']:.2f}  |  "
              f"MDD: {best['_max_drawdown']:.2f}%  |  Sharpe: {best['_sharpe']:.2f}")
        print(f"    IS    : {best['_composite']:+.2f}  |  "
              f"WF: {best.get('_wf_score', -999):+.2f}  |  "
              f"OOS: {best.get('_oos_score', -999):+.2f}  |  "
              f"Final: {best.get('_final_score', -999):+.2f}")

        print(f"\n  📋 Snippet — copia in Config():")
        print(f"  cfg = Config(")
        for k in ["ema_fast", "ema_slow", "ema_trend",
                  "adx_period", "adx_min",
                  "rsi_period", "rsi_ob", "rsi_os",
                  "atr_period", "atr_stop_mult", "atr_tp_mult",
                  "bb_period", "bb_std", "vol_mult",
                  "signal_threshold", "position_pct"]:
            if k in best:
                print(f"      {k.upper():<25} = {best[k]},")
        print(f"  )")
        print(f"{'═'*W}")

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _progress(i: int, total: int, found: int, t0: float):
        pct = i / total * 100
        eta = (time.time() - t0) / i * (total - i) if i > 0 else 0
        print(f"    [{pct:4.0f}%] {found} valide | ETA {eta:.0f}s    ", end="\r")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Strategy Optimizer v2.0 — Multi-interval + Walk-Forward",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python optimizer.py --random 500
  python optimizer.py --random 1000 --intervals 1h,4h,1d --days 730
  python optimizer.py --random 2000 --wf-splits 4 --symbols BTCUSDC,ETHUSDC
  python optimizer.py --grid   --intervals 1d --days 730
        """)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--random", type=int, metavar="N",
                      help="Latin Hypercube Sampling: N combinazioni")
    mode.add_argument("--grid",   action="store_true",
                      help="Grid search: tutte le combinazioni valide")

    parser.add_argument("--symbols",    default=",".join(DEFAULT_SYMBOLS),
                        help="Simboli separati da virgola")
    parser.add_argument("--intervals",  default="1h,4h,1d",
                        help="Intervalli da testare (default: 1h,4h,1d)")
    parser.add_argument("--days",       type=int,   default=730,
                        help="Giorni di storico (default: 730)")
    parser.add_argument("--min-trades", type=int,   default=10,
                        help="Trade minimi per considerare una config (default: 10)")
    parser.add_argument("--jobs",       type=int,
                        default=max(1, min(cpu_count() - 1, 3)),
                        help="Job paralleli — max 3 per RPi4 (default: auto)")
    parser.add_argument("--top",        type=int,   default=10,
                        help="Posizioni leaderboard da mostrare (default: 10)")
    parser.add_argument("--wf-splits",  type=int,   default=3,
                        help="Finestre walk-forward IS (default: 3)")
    parser.add_argument("--out",        default="optimizer",
                        help="Prefisso output file (default: optimizer)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--force-download", action="store_true",
                        help="Forza re-download ignorando la cache")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    symbols   = [s.strip().upper() for s in args.symbols.split(",")]
    intervals = [iv.strip() for iv in args.intervals.split(",")]

    # Stima dimensione search space
    space_size = 1
    for v in SEARCH_SPACE.values():
        space_size *= len(v)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  CRYPTO STRATEGY OPTIMIZER v2.0                             ║
║  Multi-interval · Walk-Forward · Latin Hypercube · RPi4     ║
╠══════════════════════════════════════════════════════════════╣
║  Simboli   : {', '.join(symbols[:4])+'...' if len(symbols)>4 else ', '.join(symbols):<43} ║
║  Intervalli: {', '.join(intervals):<47} ║
║  Periodo   : {args.days} giorni  [{100*(1-OOS_FRACTION):.0f}% IS / {OOS_FRACTION*100:.0f}% OOS]{'':<18} ║
║  WF splits : {args.wf_splits}  |  Jobs: {args.jobs} (RPi4 safe){'':<28} ║
║  Spazio    : ~{space_size:,} combo totali{'':<31} ║
╚══════════════════════════════════════════════════════════════╝""")

    opt = Optimizer(
        symbols    = symbols,
        days       = args.days,
        min_trades = args.min_trades,
        n_jobs     = args.jobs,
        top_n      = args.top,
        wf_splits  = args.wf_splits,
        out_prefix = args.out,
    )

    # Genera param list UNA VOLTA — riusata per tutti gli interval
    if args.random:
        n = args.random
        print(f"\n  Generazione {n} param set [Latin Hypercube, seed={args.seed}]...")
        param_list = lhs_params(n)
        param_list.append(ParamSet())     # aggiunge config default come baseline
        print(f"  {len(param_list)} param set pronti.")
    else:
        print(f"\n  Generazione grid search...")
        param_list = grid_params()
        n = len(param_list)
        print(f"  {n:,} combinazioni valide (su ~{space_size:,} totali).")
        if n > 15_000:
            print(f"  ⚠️  Grid grande ({n:,} combo)! --random 2000 è più veloce.")
            ans = input("  Procedere? [s/N]: ").strip().lower()
            if ans != "s":
                sys.exit(0)

    # Pre-download dati per tutti gli interval (sfrutta la cache)
    print(f"\n{'─'*68}")
    print("  Pre-download dati per tutti gli intervalli...")
    for iv in intervals:
        opt.load_data(iv, force=args.force_download)

    # ── Ottimizzazione per ogni interval ─────────────────────────────────────
    all_best: Dict[str, dict] = {}

    for iv in intervals:
        sep = "═" * 68
        print(f"\n\n{sep}")
        print(f"  ▶  OTTIMIZZAZIONE  [{iv}]")
        print(sep)

        results = opt.run_interval(iv, param_list)
        if not results:
            continue

        opt.save(results, iv)
        opt.print_leaderboard(results, iv)

        best = results[0]
        all_best[iv] = {
            **{k: v for k, v in best.items() if not k.startswith("_")},
            "_metrics": {
                "n_trades":     best["_n_trades"],
                "win_rate":     round(best["_win_rate"],    2),
                "total_return": round(best["_total_return"],2),
                "sharpe":       round(best["_sharpe"],      3),
                "profit_factor":round(best["_profit_factor"],2),
                "max_drawdown": round(best["_max_drawdown"],2),
                "is_score":     round(best["_composite"],   3),
                "wf_score":     round(best.get("_wf_score",  -999), 3),
                "oos_score":    round(best.get("_oos_score", -999), 3),
                "final_score":  round(best.get("_final_score",-999),3),
            },
        }

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    sep = "═" * 68
    print(f"\n\n{sep}")
    print(f"  🏁  RIEPILOGO — BEST CONFIG PER INTERVAL")
    print(sep)
    print(f"  {'Interval':<10}  {'Final':>7}  {'WF':>7}  {'OOS':>7}  "
          f"{'WR%':>6}  {'Ret%':>7}  {'Sharpe':>7}  {'Trades':>6}")
    print(f"  {'─'*66}")

    for iv, cfg in all_best.items():
        m = cfg["_metrics"]
        print(
            f"  {iv:<10}  {m['final_score']:>+7.2f}  {m['wf_score']:>+7.2f}"
            f"  {m['oos_score']:>+7.2f}  {m['win_rate']:>6.1f}"
            f"  {m['total_return']:>+7.2f}  {m['sharpe']:>+7.2f}"
            f"  {m['n_trades']:>6}"
        )

    # Salva JSON riepilogo globale
    summary_path = "best_configs_all.json"
    with open(summary_path, "w") as f:
        json.dump(all_best, f, indent=2)

    print(f"\n  💾 Riepilogo globale: {summary_path}")
    print(f"\n  Output per interval:")
    for iv in all_best:
        print(f"    [{iv}]  optimizer_{iv}.csv  |  best_config_{iv}.json")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()