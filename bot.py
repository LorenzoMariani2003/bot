"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           CRYPTO TRADING BOT v2.1 — QUANT ARCHITECTURE                     ║
║                                                                              ║
║  Modalità di esecuzione:                                                    ║
║    --live       → trading reale su Binance (richiede API key)               ║
║    --livetest   → paper trading live (portafoglio virtuale, dati reali)     ║
║    --backtest   → backtest vettoriale su storico                            ║
║                                                                              ║
║  LIVETEST: stessa logica del live, ma nessun ordine reale viene piazzato.  ║
║    Gli acquisti/vendite virtuali usano i prezzi di mercato reali in tempo  ║
║    reale, simulando commissioni (0.1%) e slippage (0.02%). Permette di      ║
║    validare la configurazione uscita dall'optimizer senza rischiare        ║
║    capitale reale.                                                           ║
║                                                                              ║
║  DISCLAIMER: Solo scopo educativo. Il trading crypto è ad alto rischio.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import asyncio
import json
import logging
import math
import os
import random
import sys
import time
import csv
import threading
import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from binance import AsyncClient, BinanceSocketManager
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
    _HAS_BINANCE = True
except ImportError:
    _HAS_BINANCE = False
    print("WARN: python-binance non installato. Avvia: pip install python-binance")


# ══════════════════════════════════════════════════════════════════════════════
#  1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Tutti i parametri del bot. Modifica qui senza toccare il resto."""

    # ── Credenziali ──────────────────────────────────────────────────────────
    API_KEY:    str = ""
    API_SECRET: str = ""
    TELEGRAM_TOKEN:   str = ""
    TELEGRAM_CHAT_ID: str = ""

    # ── Asset da tradare ──────────────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "BTCUSDC", "ETHUSDC", "BNBUSDC", "SOLUSDC",
        "ADAUSDC", "XRPUSDC", "DOGEUSDC", "AVAXUSDC",
    ])
    QUOTE_ASSET: str = "USDC"
    INTERVAL:   str = "1h"
    CANDLE_LIMIT: int = 300

    # ── Gestione del rischio ─────────────────────────────────────────────────
    MAX_OPEN_POSITIONS: int   = 5
    KELLY_FRACTION:     float = 0.25
    MIN_KELLY_BET:      float = 0.02
    MAX_KELLY_BET:      float = 0.20
    MIN_USDC_TRADE:     float = 11.0
    DAILY_DRAWDOWN_LIMIT: float = 0.05

    # ── ATR Trailing Stop ────────────────────────────────────────────────────
    ATR_PERIOD:      int   = 14
    ATR_STOP_MULT:   float = 2.0
    ATR_TP_MULT:     float = 3.0

    # ── Parametri indicatori ─────────────────────────────────────────────────
    EMA_FAST:    int   = 16
    EMA_SLOW:    int   = 21
    EMA_TREND:   int   = 200
    ADX_PERIOD:  int   = 10
    ADX_MIN:     float = 22.0
    RSI_PERIOD:  int   = 14
    RSI_OB:      float = 70.0
    RSI_OS:      float = 30.0
    BB_PERIOD:   int   = 20
    BB_STD:      float = 2.0

    # ── Scoring segnali ──────────────────────────────────────────────────────
    SIGNAL_THRESHOLD: float = 60.0
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "ema_trend":      25.0,
        "ema_cross":      20.0,
        "adx_strength":   15.0,
        "rsi_ok":         15.0,
        "bb_position":    10.0,
        "volume_spike":   10.0,
        "ob_imbalance":    5.0,
    })

    # ── Order Book ───────────────────────────────────────────────────────────
    OB_DEPTH_LEVELS: int   = 10
    OB_IMBALANCE_TH: float = 0.55

    # ── Slippage control ─────────────────────────────────────────────────────
    MAX_SLIPPAGE_PCT: float = 0.003

    # ── Loop timing ──────────────────────────────────────────────────────────
    LOOP_SLEEP_SEC:   int = 60
    WS_RECONNECT_SEC: int = 5

    # ── File paths (live) ────────────────────────────────────────────────────
    LOG_FILE:    str = "bot_v2.log"
    TRADE_LOG:   str = "trades_v2.csv"
    STATE_FILE:  str = "state_v2.json"
    STATUS_FILE: str = "status_v2.json"

    # ── File paths (livetest) ────────────────────────────────────────────────
    LT_LOG_FILE:   str = "bot_livetest.log"
    LT_TRADE_LOG:  str = "trades_livetest.csv"
    LT_STATE_FILE: str = "state_livetest.json"
    LT_VP_FILE:    str = "vp_livetest.json"     # stato portafoglio virtuale

    # ── Livetest reporting ───────────────────────────────────────────────────
    LT_REPORT_INTERVAL: int = 10    # ogni N cicli stampa/invia report P&L


# ══════════════════════════════════════════════════════════════════════════════
#  2. LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def build_logger(log_file: str, level=logging.INFO) -> logging.Logger:
    """Logger con rotazione file (max 5 MB × 3 backup). Thread-safe."""
    logger = logging.getLogger(log_file)   # nome unico per live vs livetest
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
#  3. INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """Calcolo vettoriale puro con NumPy/Pandas."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high, low, close = df["high"], df["low"], df["close"]
        prev_high = high.shift(1)
        prev_low  = low.shift(1)
        dm_plus   = (high - prev_high).clip(lower=0)
        dm_minus  = (prev_low - low).clip(lower=0)
        mask      = dm_plus < dm_minus
        dm_plus[mask] = 0
        mask2     = dm_minus < dm_plus
        dm_minus[mask2] = 0
        atr_val   = IndicatorEngine.atr(df, period)
        di_plus   = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_val
        di_minus  = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_val
        dx        = (100 * (di_plus - di_minus).abs() /
                     (di_plus + di_minus).replace(0, np.nan))
        adx_val   = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx_val, di_plus, di_minus

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int,
                        std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = series.rolling(period).mean()
        std    = series.rolling(period).std()
        return middle + std_mult * std, middle, middle - std_mult * std

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26,
             signal=9) -> Tuple[pd.Series, pd.Series]:
        ema_f       = series.ewm(span=fast,   adjust=False).mean()
        ema_s       = series.ewm(span=slow,   adjust=False).mean()
        macd_line   = ema_f - ema_s
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def populate(self, df: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
        df    = df.copy()
        close = df["close"]
        df["ema_fast"]  = self.ema(close, cfg.EMA_FAST)
        df["ema_slow"]  = self.ema(close, cfg.EMA_SLOW)
        df["ema_trend"] = self.ema(close, cfg.EMA_TREND)
        df["rsi"]       = self.rsi(close, cfg.RSI_PERIOD)
        df["atr"]       = self.atr(df,    cfg.ATR_PERIOD)
        adx_val, di_plus, di_minus = self.adx(df, cfg.ADX_PERIOD)
        df["adx"]       = adx_val
        df["di_plus"]   = di_plus
        df["di_minus"]  = di_minus
        bb_up, bb_mid, bb_low = self.bollinger_bands(close, cfg.BB_PERIOD, cfg.BB_STD)
        df["bb_upper"]  = bb_up
        df["bb_middle"] = bb_mid
        df["bb_lower"]  = bb_low
        macd_line, signal_line = self.macd(close)
        df["macd"]        = macd_line
        df["macd_signal"] = signal_line
        vol_ma            = df["volume"].rolling(20).mean()
        df["vol_spike"]   = df["volume"] > (vol_ma * 1.5)
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    score:    float     = 0.0
    buy:      bool      = False
    sell:     bool      = False
    reasons:  List[str] = field(default_factory=list)
    atr:      float     = 0.0
    sl_price: float     = 0.0
    tp_price: float     = 0.0


class Strategy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ind = IndicatorEngine()

    def score_buy(self, df: pd.DataFrame, ob_imbalance: float = 0.5) -> SignalResult:
        cfg    = self.cfg
        result = SignalResult()
        if len(df) < cfg.EMA_TREND + 5:
            result.reasons.append("Dati insufficienti")
            return result

        last  = df.iloc[-1]
        score = 0.0
        w     = cfg.WEIGHTS

        if last["close"] > last["ema_trend"]:
            score += w["ema_trend"]
            pct    = (last["close"] / last["ema_trend"] - 1) * 100
            result.reasons.append(f"✓ EMA trend: +{pct:.2f}% sopra EMA{cfg.EMA_TREND}")
        else:
            result.reasons.append(f"✗ EMA trend: sotto EMA{cfg.EMA_TREND}")

        if last["ema_fast"] > last["ema_slow"]:
            score += w["ema_cross"]
            result.reasons.append(f"✓ EMA cross: EMA{cfg.EMA_FAST} > EMA{cfg.EMA_SLOW}")
        else:
            result.reasons.append(f"✗ EMA cross: EMA{cfg.EMA_FAST} < EMA{cfg.EMA_SLOW}")

        if last["adx"] > cfg.ADX_MIN and last["di_plus"] > last["di_minus"]:
            score += w["adx_strength"]
            result.reasons.append(f"✓ ADX={last['adx']:.1f} (trend forte)")
        else:
            result.reasons.append(f"✗ ADX={last['adx']:.1f} (trend debole)")

        rsi_val = last["rsi"]
        if rsi_val < cfg.RSI_OB:
            partial = w["rsi_ok"] * (1 - max(0, rsi_val - cfg.RSI_OS) /
                                     (cfg.RSI_OB - cfg.RSI_OS))
            score  += partial
            result.reasons.append(f"✓ RSI={rsi_val:.1f} (zona OK)")
        else:
            result.reasons.append(f"✗ RSI={rsi_val:.1f} (overbought)")

        bb_pos = (last["close"] - last["bb_lower"]) / (
                  last["bb_upper"] - last["bb_lower"] + 1e-9)
        if bb_pos < 0.5:
            score += w["bb_position"] * (1 - bb_pos * 2)
            result.reasons.append(f"✓ BB: {bb_pos*100:.0f}% della banda")
        else:
            result.reasons.append(f"✗ BB: {bb_pos*100:.0f}% della banda")

        if last["vol_spike"]:
            score += w["volume_spike"]
            result.reasons.append("✓ Volume spike (> 1.5× media)")
        else:
            result.reasons.append("✗ Volume nella norma")

        if ob_imbalance > cfg.OB_IMBALANCE_TH:
            score += w["ob_imbalance"]
            result.reasons.append(f"✓ OB Imbalance={ob_imbalance:.2f}")
        else:
            result.reasons.append(f"✗ OB Imbalance={ob_imbalance:.2f}")

        atr_val         = last["atr"]
        result.atr      = atr_val
        result.sl_price = last["close"] - cfg.ATR_STOP_MULT * atr_val
        result.tp_price = last["close"] + cfg.ATR_TP_MULT  * atr_val
        result.score    = round(score, 2)
        result.buy      = score >= cfg.SIGNAL_THRESHOLD
        return result

    def check_sell(self, df: pd.DataFrame, position: dict) -> Tuple[bool, str]:
        if len(df) < 3:
            return False, ""
        last  = df.iloc[-1]
        price = last["close"]
        sl    = position.get("trailing_sl", position.get("sl_price", 0))
        if sl and price <= sl:
            return True, f"🛑 Trailing Stop Loss: {price:.4f} ≤ {sl:.4f}"
        tp = position.get("tp_price", 0)
        if tp and price >= tp:
            return True, f"🎯 Take Profit: {price:.4f} ≥ {tp:.4f}"
        if last["ema_fast"] < last["ema_slow"]:
            if last["adx"] < self.cfg.ADX_MIN or last["di_minus"] > last["di_plus"]:
                return True, "📉 EMA cross ribassista + ADX debole"
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#  5. RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg    = cfg
        self.log    = logger
        self._daily_start_equity: Optional[float] = None
        self._daily_date:         Optional[str]   = None
        self.trading_halted:      bool             = False

    def kelly_position_size(self, win_rate: float, equity: float,
                             atr: float, entry_price: float) -> float:
        cfg       = self.cfg
        R         = cfg.ATR_TP_MULT / cfg.ATR_STOP_MULT
        L         = 1 - win_rate
        numerator = win_rate * R - L
        if numerator <= 0:
            self.log.warning(f"Kelly negativo (W={win_rate:.2f}): skip")
            return 0.0
        f_star   = numerator / R
        f_scaled = f_star * cfg.KELLY_FRACTION
        f_final  = max(cfg.MIN_KELLY_BET, min(cfg.MAX_KELLY_BET, f_scaled))
        amount   = equity * f_final
        self.log.info(f"Kelly: W={win_rate:.2f} R={R:.2f} f={f_final:.3f} → {amount:.2f} USDC")
        return amount

    def update_trailing_stop(self, position: dict, current_price: float) -> dict:
        atr        = position.get("atr", 0)
        new_sl     = current_price - self.cfg.ATR_STOP_MULT * atr
        current_sl = position.get("trailing_sl", position.get("sl_price", 0))
        if new_sl > current_sl:
            position["trailing_sl"] = new_sl
        return position

    def check_daily_drawdown(self, current_equity: float) -> bool:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._daily_date != today:
            self._daily_date         = today
            self._daily_start_equity = current_equity
            self.trading_halted      = False
            self.log.info(f"Nuovo giorno: equity iniziale = {current_equity:.2f} USDC")
        if self._daily_start_equity and self._daily_start_equity > 0:
            dd = (self._daily_start_equity - current_equity) / self._daily_start_equity
            if dd >= self.cfg.DAILY_DRAWDOWN_LIMIT:
                if not self.trading_halted:
                    self.trading_halted = True
                    self.log.error(
                        f"⚠️ DAILY DRAWDOWN -{dd*100:.2f}% raggiunto. Trading BLOCCATO.")
                return True
        return False

    def estimate_win_rate(self, trade_log: List[dict]) -> float:
        closed = [t for t in trade_log if t.get("pnl") is not None][-50:]
        if len(closed) < 5:
            return 0.50
        wins = sum(1 for t in closed if float(t["pnl"]) > 0)
        return wins / len(closed)


# ══════════════════════════════════════════════════════════════════════════════
#  6. BINANCE CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class BinanceClient:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg    = cfg
        self.log    = logger
        self.client: Optional[Client] = None
        self._request_times: deque    = deque(maxlen=1200)

    def connect(self, paper_only: bool = False):
        if not _HAS_BINANCE:
            raise RuntimeError("python-binance non installato")
        # Con chiavi vuote funziona comunque per endpoint pubblici
        self.client = Client(self.cfg.API_KEY, self.cfg.API_SECRET)
        self.client.session.timeout = 30
        mode = " (sola lettura — livetest)" if paper_only else ""
        self.log.info(f"Connessione Binance stabilita{mode}")

    def _rate_check(self):
        now = time.time()
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        if len(self._request_times) >= 1100:
            sleep_time = 60 - (now - self._request_times[0]) + 1
            self.log.warning(f"Rate limit: attendo {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._request_times.append(time.time())

    def request(self, func, *args, max_retries=4, **kwargs):
        delay = 1.0
        for attempt in range(max_retries):
            try:
                self._rate_check()
                return func(*args, **kwargs)
            except (BinanceRequestException,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    jitter = random.uniform(0, delay * 0.5)
                    wait   = delay + jitter
                    self.log.warning(f"[{func.__name__}] retry {attempt+1}: {e}. Wait {wait:.1f}s")
                    time.sleep(wait)
                    delay *= 2
                else:
                    self.log.error(f"[{func.__name__}] tutti i retry falliti: {e}")
                    raise
            except BinanceAPIException as e:
                if e.code in (-1003, -1015):
                    time.sleep(60)
                    continue
                raise
        return None

    def get_klines(self, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
        raw = self.request(self.client.get_klines,
                           symbol=symbol, interval=interval, limit=limit)
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open_time", "open", "high", "low", "close", "volume"]]

    def get_ticker_price(self, symbol: str) -> float:
        ticker = self.request(self.client.get_symbol_ticker, symbol=symbol)
        return float(ticker["price"]) if ticker else 0.0

    def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        return self.request(self.client.get_order_book, symbol=symbol, limit=limit) or {}

    def ob_imbalance(self, symbol: str) -> float:
        book    = self.get_order_book(symbol, limit=self.cfg.OB_DEPTH_LEVELS * 2)
        if not book:
            return 0.5
        n       = self.cfg.OB_DEPTH_LEVELS
        bid_vol = sum(float(b[1]) for b in book.get("bids", [])[:n])
        ask_vol = sum(float(a[1]) for a in book.get("asks", [])[:n])
        total   = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def get_usdc_balance(self) -> float:
        bal = self.request(self.client.get_asset_balance, asset="USDC")
        return float(bal["free"]) if bal else 0.0

    def get_asset_balance(self, asset: str) -> float:
        bal = self.request(self.client.get_asset_balance, asset=asset)
        return float(bal["free"]) if bal else 0.0

    def buy_market_quote(self, symbol: str, quote_amount: float,
                         ref_price: float) -> Optional[dict]:
        current = self.get_ticker_price(symbol)
        if ref_price > 0:
            slip = abs(current - ref_price) / ref_price
            if slip > self.cfg.MAX_SLIPPAGE_PCT:
                self.log.warning(f"Slippage {symbol}: {slip*100:.3f}% > limite. Annullato.")
                return None
        q = float(Decimal(str(quote_amount)).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN))
        if q < self.cfg.MIN_USDC_TRADE:
            self.log.warning(f"Quote {q} < minimo {self.cfg.MIN_USDC_TRADE}")
            return None
        return self.request(self.client.create_order,
                            symbol=symbol, side="BUY", type="MARKET", quoteOrderQty=q)

    def sell_market_qty(self, symbol: str, qty: float) -> Optional[dict]:
        info = self.request(self.client.get_symbol_info, symbol)
        if not info:
            return None
        step     = next((float(f["stepSize"])
                         for f in info["filters"] if f["filterType"] == "LOT_SIZE"), 0.001)
        decimals = max(0, -int(math.floor(math.log10(step))))
        adj_qty  = float(
            (Decimal(str(qty)) / Decimal(str(step))).to_integral_value(
                ROUND_DOWN) * Decimal(str(step)))
        if adj_qty <= 0:
            return None
        return self.request(self.client.create_order,
                            symbol=symbol, side="SELL", type="MARKET",
                            quantity=round(adj_qty, decimals))


# ══════════════════════════════════════════════════════════════════════════════
#  7. POSITION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class PositionManager:
    def __init__(self, cfg: Config, logger: logging.Logger,
                 state_file: Optional[str] = None,
                 trade_log:  Optional[str] = None):
        self.cfg       = cfg
        self.log       = logger
        self.state_file = state_file or cfg.STATE_FILE
        self.trade_log  = trade_log  or cfg.TRADE_LOG
        self.positions: Dict[str, Optional[dict]] = {}
        self.trade_log_list: List[dict] = []

    def load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.positions = state.get("positions", {})
                self.log.info(f"Stato caricato da {self.state_file}")
            except Exception as e:
                self.log.error(f"Errore caricamento stato: {e}")
        if os.path.exists(self.trade_log):
            try:
                with open(self.trade_log, newline="") as f:
                    self.trade_log_list = list(csv.DictReader(f))
            except Exception:
                pass

    def save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"ts": datetime.utcnow().isoformat(),
                           "positions": self.positions}, f, indent=2)
        except Exception as e:
            self.log.error(f"Errore salvataggio stato: {e}")

    def open_position(self, symbol: str, entry_price: float, qty: float,
                      invested: float, sl: float, tp: float, atr: float):
        self.positions[symbol] = {
            "entry_price":  entry_price,
            "qty":          qty,
            "invested":     invested,
            "sl_price":     sl,
            "trailing_sl":  sl,
            "tp_price":     tp,
            "atr":          atr,
            "open_time":    datetime.utcnow().isoformat(),
        }
        self._csv_log(symbol, "BUY", entry_price, qty, invested)
        self.save()

    def close_position(self, symbol: str, exit_price: float, qty: float, reason: str) -> float:
        pos = self.positions.get(symbol)
        pnl = 0.0
        if pos:
            pnl     = (exit_price - pos["entry_price"]) * qty
            pnl_pct = pnl / pos["invested"] * 100 if pos["invested"] else 0
            self._csv_log(symbol, "SELL", exit_price, qty, pos["invested"], pnl, pnl_pct)
            self.trade_log_list.append({
                "symbol": symbol, "pnl": pnl,
                "pnl_pct": pnl_pct, "reason": reason
            })
            self.log.info(
                f"CLOSE {symbol} @ {exit_price:.4f} | "
                f"PnL={pnl:+.2f} ({pnl_pct:+.2f}%) | {reason}")
        self.positions[symbol] = None
        self.save()
        return pnl

    def _csv_log(self, symbol, action, price, qty, invested,
                 pnl=None, pnl_pct=None):
        entry  = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol, "action": action,
            "price": price, "qty": qty, "invested": invested,
            "pnl": pnl or 0, "pnl_pct": pnl_pct or 0,
        }
        exists = os.path.exists(self.trade_log)
        try:
            with open(self.trade_log, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=entry.keys())
                if not exists:
                    w.writeheader()
                w.writerow(entry)
        except Exception as e:
            self.log.error(f"CSV log error: {e}")

    def open_count(self) -> int:
        return sum(1 for v in self.positions.values() if v)

    def is_open(self, symbol: str) -> bool:
        return bool(self.positions.get(symbol))

    def get(self, symbol: str) -> Optional[dict]:
        return self.positions.get(symbol)

    def set(self, symbol: str, pos: dict):
        self.positions[symbol] = pos
        self.save()

    def stats(self) -> dict:
        closed = [t for t in self.trade_log_list if "pnl" in t]
        if not closed:
            return {"total": 0, "wins": 0, "total_pnl": 0.0,
                    "win_rate": 0.5, "avg_pnl": 0.0}
        pnls = [float(t["pnl"]) for t in closed]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "total":     len(pnls),
            "wins":      wins,
            "total_pnl": sum(pnls),
            "win_rate":  wins / len(pnls),
            "avg_pnl":   sum(pnls) / len(pnls),
            "best":      max(pnls),
            "worst":     min(pnls),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  8. TELEGRAM HANDLER
# ══════════════════════════════════════════════════════════════════════════════

class TelegramHandler:
    def __init__(self, cfg: Config, logger: logging.Logger,
                 pm: PositionManager, bc: BinanceClient,
                 vp: Optional["VirtualPortfolio"] = None):
        self.cfg             = cfg
        self.log             = logger
        self.pm              = pm
        self.bc              = bc
        self.vp              = vp          # None in modalità live
        self.trading_enabled = True
        self.start_time      = datetime.utcnow()
        self._last_update_id = 0

    def send(self, text: str):
        if not self.cfg.TELEGRAM_TOKEN or not self.cfg.TELEGRAM_CHAT_ID:
            return
        url     = f"https://api.telegram.org/bot{self.cfg.TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": self.cfg.TELEGRAM_CHAT_ID,
                   "text": text, "parse_mode": "Markdown"}
        try:
            requests.post(url, data=payload, timeout=10)
        except Exception:
            try:
                payload.pop("parse_mode")
                requests.post(url, data=payload, timeout=10)
            except Exception:
                pass

    def _get_updates(self) -> list:
        if not self.cfg.TELEGRAM_TOKEN:
            return []
        url = f"https://api.telegram.org/bot{self.cfg.TELEGRAM_TOKEN}/getUpdates"
        try:
            r = requests.get(url, params={"timeout": 1,
                                           "offset": self._last_update_id + 1},
                             timeout=5)
            if r.status_code == 200:
                return r.json().get("result", [])
        except Exception:
            pass
        return []

    def _uptime(self) -> str:
        d = datetime.utcnow() - self.start_time
        h, r = divmod(d.seconds, 3600)
        m, _ = divmod(r, 60)
        return f"{d.days}d {h}h {m}m" if d.days else (
               f"{h}h {m}m" if h else f"{m}m")

    def handle(self, cmd: str) -> str:
        cmd = cmd.strip()
        low = cmd.lower()

        if low in ("/status", "/info", "status", "info"):
            st     = self.pm.stats()
            is_lt  = self.vp is not None
            prefix = "📝 *[LIVETEST]* " if is_lt else ""

            if is_lt:
                # Calcola mark-to-market delle posizioni virtuali aperte
                prices = {}
                for sym in self.pm.positions:
                    if self.pm.positions[sym]:
                        try:
                            prices[sym] = self.bc.get_ticker_price(sym)
                        except Exception:
                            pass
                equity   = self.vp.mark_to_market(self.pm.positions, prices)
                balance  = self.vp.balance
                ret_pct  = (equity - self.vp.initial_capital) / self.vp.initial_capital * 100
                ts       = "🟢 ATTIVO" if self.trading_enabled else "🔴 SOSPESO"
                return (
                    f"{prefix}*BOT v2.1 STATUS*\n\n"
                    f"⏰ Uptime: {self._uptime()}\n"
                    f"🎛️ Trading: {ts}\n\n"
                    f"💰 Capitale iniziale: {self.vp.initial_capital:.2f} USDC\n"
                    f"💵 Cash virtuale:     {balance:.2f} USDC\n"
                    f"📊 Equity (MTM):      {equity:.2f} USDC\n"
                    f"{'🟢' if ret_pct >= 0 else '🔴'} Rendimento:        {ret_pct:+.2f}%\n"
                    f"📈 Posizioni aperte: {self.pm.open_count()}/{self.cfg.MAX_OPEN_POSITIONS}\n\n"
                    f"📋 Trade chiuse: {st['total']}\n"
                    f"🎯 Win rate: {st['win_rate']*100:.1f}%\n"
                    f"💎 PnL totale: {st['total_pnl']:+.2f} USDC\n"
                    f"🚀 Miglior trade: {st.get('best',0):+.2f} USDC\n"
                    f"📉 Peggior trade: {st.get('worst',0):+.2f} USDC"
                )
            else:
                equity = self.bc.get_usdc_balance()
                ts     = "🟢 ATTIVO" if self.trading_enabled else "🔴 SOSPESO"
                return (
                    f"🤖 *BOT v2.1 STATUS*\n\n"
                    f"⏰ Uptime: {self._uptime()}\n"
                    f"🎛️ Trading: {ts}\n"
                    f"💰 USDC libero: {equity:.2f}\n"
                    f"📈 Posizioni aperte: {self.pm.open_count()}/{self.cfg.MAX_OPEN_POSITIONS}\n\n"
                    f"📋 Trade chiuse: {st['total']}\n"
                    f"🎯 Win rate: {st['win_rate']*100:.1f}%\n"
                    f"💎 PnL totale: {st['total_pnl']:+.2f} USDC\n"
                    f"🚀 Miglior trade: {st.get('best',0):+.2f} USDC\n"
                    f"📉 Peggior trade: {st.get('worst',0):+.2f} USDC"
                )

        if low in ("/stop", "stop", "/pause", "pause"):
            self.trading_enabled = False
            return "🔴 Trading SOSPESO. Posizioni aperte rimangono attive."

        if low in ("/start", "start", "/resume", "resume"):
            self.trading_enabled = True
            return "🟢 Trading RIATTIVATO."

        if low in ("/positions", "positions"):
            open_pos = {k: v for k, v in self.pm.positions.items() if v}
            if not open_pos:
                return "📈 Nessuna posizione aperta."
            lines   = [f"{'📝 [LIVETEST] ' if self.vp else ''}📈 *POSIZIONI APERTE*\n"]
            for sym, pos in open_pos.items():
                price = self.bc.get_ticker_price(sym)
                pnl   = (price - pos["entry_price"]) * pos["qty"]
                pct   = pnl / pos["invested"] * 100 if pos["invested"] else 0
                lines.append(
                    f"*{sym}*: entry={pos['entry_price']:.4f} "
                    f"now={price:.4f} PnL={pnl:+.2f} ({pct:+.1f}%)\n"
                    f"  SL={pos.get('trailing_sl',0):.4f} "
                    f"TP={pos.get('tp_price',0):.4f}")
            return "\n".join(lines)

        if low in ("/reset", "reset") and self.vp:
            # Comando speciale livetest: resetta portafoglio virtuale
            return ("⚠️ Per resettare il livetest, ferma il bot e cancella "
                    "state_livetest.json e vp_livetest.json")

        if low in ("/help", "help"):
            lt_cmds = "\n/reset — info reset portafoglio virtuale" if self.vp else ""
            return (
                f"{'📝 *[LIVETEST] *' if self.vp else '🤖 '}*COMANDI BOT v2.1*\n\n"
                f"/status — stato completo\n"
                f"/positions — posizioni aperte\n"
                f"/stop — sospendi nuovi acquisti\n"
                f"/start — riattiva acquisti\n"
                f"/help — questo messaggio"
                f"{lt_cmds}"
            )

        return "❓ Comando non riconosciuto. Usa /help."

    def run_forever(self):
        self.log.info("Telegram handler avviato")
        while True:
            try:
                updates = self._get_updates()
                for upd in updates:
                    self._last_update_id = upd.get("update_id", 0)
                    msg     = upd.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id == self.cfg.TELEGRAM_CHAT_ID and "text" in msg:
                        text  = msg["text"]
                        self.log.info(f"Telegram cmd: {text}")
                        reply = self.handle(text)
                        self.send(reply)
            except Exception as e:
                self.log.error(f"Telegram error: {e}")
            time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
#  9. VIRTUAL PORTFOLIO  (usato solo in --livetest)
# ══════════════════════════════════════════════════════════════════════════════

class VirtualPortfolio:
    """
    Portafoglio virtuale per il paper trading in modalità --livetest.

    Tiene traccia del cash virtuale (USDC), simula commissioni e slippage
    su ogni operazione e calcola l'equity mark-to-market includendo le
    posizioni aperte valutate ai prezzi correnti di mercato.

    Persiste il proprio stato in un file JSON separato dal PositionManager,
    così balance e capital iniziale sopravvivono a riavvii del bot.
    """

    COMMISSION_RATE = 0.001   # 0.1% per lato (come Binance con BNB)
    SLIPPAGE_RATE   = 0.0002  # 0.02% slippage simulato per lato

    def __init__(self, initial_capital: float, vp_file: str,
                 logger: logging.Logger):
        self.initial_capital = initial_capital
        self.balance         = initial_capital   # cash USDC disponibile
        self.vp_file         = vp_file
        self.log             = logger

    # ── Operazioni ────────────────────────────────────────────────────────────

    def simulate_buy(self, price: float, invest: float) -> Tuple[float, float]:
        """
        Simula acquisto al prezzo corrente con commissione e slippage.
        Ritorna (qty_acquistata, entry_price_effettivo).

        Il balance viene decrementato di `invest` USDC.
        """
        entry_price = price * (1 + self.SLIPPAGE_RATE)
        commission  = invest * self.COMMISSION_RATE
        qty         = (invest - commission) / entry_price
        self.balance -= invest
        self.log.info(
            f"[VT] BUY simulato: invest={invest:.2f} qty={qty:.6f} "
            f"@ {entry_price:.4f} (comm={commission:.3f})")
        return qty, entry_price

    def simulate_sell(self, price: float, qty: float,
                      invested: float) -> Tuple[float, float]:
        """
        Simula vendita al prezzo corrente con commissione e slippage.
        Ritorna (pnl_netto, proceeds_netti).

        Il balance viene incrementato dei proceeds netti.
        """
        exit_price = price * (1 - self.SLIPPAGE_RATE)
        gross      = exit_price * qty
        commission = gross * self.COMMISSION_RATE
        proceeds   = gross - commission
        pnl        = proceeds - invested
        self.balance += proceeds
        self.log.info(
            f"[VT] SELL simulato: qty={qty:.6f} @ {exit_price:.4f} "
            f"pnl={pnl:+.2f} USDC (comm={commission:.3f})")
        return pnl, proceeds

    # ── Metriche ──────────────────────────────────────────────────────────────

    def mark_to_market(self, positions: Dict[str, Optional[dict]],
                       current_prices: Dict[str, float]) -> float:
        """
        Equity totale = cash + valore mark-to-market di tutte le posizioni aperte.
        Se il prezzo di un simbolo non è disponibile, usa il prezzo di entrata.
        """
        equity = self.balance
        for sym, pos in positions.items():
            if not pos:
                continue
            price   = current_prices.get(sym, pos["entry_price"])
            mkt_val = price * pos["qty"]
            equity += mkt_val
        return equity

    def performance(self, positions: Dict[str, Optional[dict]],
                    current_prices: Dict[str, float]) -> dict:
        """Ritorna un dict con le principali metriche di performance."""
        equity  = self.mark_to_market(positions, current_prices)
        ret_pct = (equity - self.initial_capital) / self.initial_capital * 100
        return {
            "initial_capital": self.initial_capital,
            "cash_balance":    self.balance,
            "total_equity":    equity,
            "return_pct":      ret_pct,
            "open_positions":  sum(1 for v in positions.values() if v),
        }

    # ── Persistenza ───────────────────────────────────────────────────────────

    def save(self):
        try:
            data = {
                "initial_capital": self.initial_capital,
                "balance":         self.balance,
                "updated_at":      datetime.utcnow().isoformat(),
            }
            with open(self.vp_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log.error(f"VirtualPortfolio save error: {e}")

    def load(self):
        if not os.path.exists(self.vp_file):
            self.log.info(f"[VT] Nessuno stato precedente — capitale iniziale: "
                          f"{self.initial_capital:.2f} USDC")
            return
        try:
            with open(self.vp_file) as f:
                data = json.load(f)
            saved_ic = data.get("initial_capital", self.initial_capital)
            self.balance = data.get("balance", self.initial_capital)
            # Avvisa se il capitale iniziale è cambiato (es. riavvio con --capital diverso)
            if abs(saved_ic - self.initial_capital) > 0.01:
                self.log.warning(
                    f"[VT] Capitale iniziale salvato ({saved_ic:.2f}) diverso "
                    f"da quello specificato ({self.initial_capital:.2f}). "
                    f"Uso quello salvato.")
                self.initial_capital = saved_ic
            self.log.info(
                f"[VT] Stato caricato: balance={self.balance:.2f} USDC | "
                f"capitale iniziale={self.initial_capital:.2f} USDC")
        except Exception as e:
            self.log.error(f"VirtualPortfolio load error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  10. TRADING BOT (ORCHESTRATORE)
# ══════════════════════════════════════════════════════════════════════════════

class TradingBot:
    """
    Orchestratore principale.

    Supporta tre modalità:
      paper_trading=False → trading reale (--live)
      paper_trading=True  → paper trading live (--livetest)

    In livetest tutti i segnali, gli indicatori e la logica di rischio sono
    identici al live; l'unica differenza è che buy/sell vengono eseguiti sul
    VirtualPortfolio invece che sull'exchange reale.
    """

    def __init__(self, cfg: Config,
                 paper_trading: bool  = False,
                 paper_capital: float = 1000.0):
        self.cfg           = cfg
        self.paper_trading = paper_trading

        # File separati per livetest vs live
        log_file   = cfg.LT_LOG_FILE   if paper_trading else cfg.LOG_FILE
        state_file = cfg.LT_STATE_FILE if paper_trading else cfg.STATE_FILE
        trade_log  = cfg.LT_TRADE_LOG  if paper_trading else cfg.TRADE_LOG

        self.log   = build_logger(log_file)
        self.bc    = BinanceClient(cfg, self.log)
        self.pm    = PositionManager(cfg, self.log,
                                     state_file=state_file,
                                     trade_log=trade_log)
        self.strat = Strategy(cfg)
        self.ind   = IndicatorEngine()
        self.rm    = RiskManager(cfg, self.log)

        # Portafoglio virtuale (solo livetest)
        self.vp: Optional[VirtualPortfolio] = None
        if paper_trading:
            self.vp = VirtualPortfolio(paper_capital, cfg.LT_VP_FILE, self.log)

        self.tg = TelegramHandler(cfg, self.log, self.pm, self.bc, self.vp)

        # Contatore cicli per report periodico livetest
        self._lt_cycle_count = 0

    # ── Avvio ─────────────────────────────────────────────────────────────────

    def start(self):
        self.bc.connect(paper_only=self.paper_trading)
        self.pm.load()
        if self.paper_trading:
            self.vp.load()

        for sym in self.cfg.SYMBOLS:
            if sym not in self.pm.positions:
                self.pm.positions[sym] = None

        if self.paper_trading:
            ret = ((self.vp.balance - self.vp.initial_capital) /
                   self.vp.initial_capital * 100)
            self.log.info(
                f"[LIVETEST] Avviato su {self.cfg.SYMBOLS} [{self.cfg.INTERVAL}] | "
                f"capitale={self.vp.initial_capital:.2f} USDC | "
                f"balance attuale={self.vp.balance:.2f} ({ret:+.2f}%)")
            self.tg.send(
                f"📝 *[LIVETEST] Bot v2.1 avviato*\n"
                f"Simboli: {', '.join(self.cfg.SYMBOLS)}\n"
                f"Intervallo: {self.cfg.INTERVAL}\n"
                f"Capitale virtuale: {self.vp.initial_capital:.2f} USDC\n"
                f"Balance attuale: {self.vp.balance:.2f} USDC ({ret:+.2f}%)\n"
                f"Usa /status per monitorare.")
        else:
            self.log.info(f"Bot avviato su {self.cfg.SYMBOLS} [{self.cfg.INTERVAL}]")
            self.tg.send(
                f"🤖 *Bot v2.1 avviato*\n"
                f"Simboli: {', '.join(self.cfg.SYMBOLS)}\n"
                f"Intervallo: {self.cfg.INTERVAL}\n"
                f"Usa /status per monitorare.")

        t = threading.Thread(target=self.tg.run_forever, daemon=True)
        t.start()

        try:
            while True:
                self._cycle()
                if self.paper_trading:
                    self.vp.save()
                time.sleep(self.cfg.LOOP_SLEEP_SEC)
        except KeyboardInterrupt:
            self.log.info("Interruzione manuale — salvo stato e uscita")
        finally:
            self.pm.save()
            if self.paper_trading:
                self.vp.save()
                report = self._livetest_report()
                print(report)
                self.tg.send(f"🛑 *[LIVETEST] Bot terminato*\n{self.tg._uptime()}\n\n"
                             + report.replace("─" * 55, ""))
            else:
                self.tg.send(f"🛑 Bot v2.1 terminato. Uptime: {self.tg._uptime()}")

    # ── Ciclo principale ──────────────────────────────────────────────────────

    def _cycle(self):
        """Un ciclo completo su tutti i simboli."""
        if self.paper_trading:
            equity = self.vp.balance
        else:
            equity = self.bc.get_usdc_balance()

        halted   = self.rm.check_daily_drawdown(equity)
        win_rate = self.rm.estimate_win_rate(self.pm.trade_log_list)

        for symbol in self.cfg.SYMBOLS:
            try:
                self._process_symbol(symbol, equity, win_rate, halted)
            except Exception as e:
                self.log.error(f"Errore su {symbol}: {e}", exc_info=True)

        # Report periodico livetest
        if self.paper_trading:
            self._lt_cycle_count += 1
            if self._lt_cycle_count % self.cfg.LT_REPORT_INTERVAL == 0:
                report = self._livetest_report()
                self.log.info(f"\n{report}")
                self.tg.send(f"📊 *[LIVETEST] Report #{self._lt_cycle_count}*\n"
                             + self._livetest_telegram_summary())

    # ── Elaborazione singolo simbolo ──────────────────────────────────────────

    def _process_symbol(self, symbol: str, equity: float,
                        win_rate: float, halted: bool):
        # 1. Candele + indicatori
        df = self.bc.get_klines(symbol, self.cfg.INTERVAL, self.cfg.CANDLE_LIMIT)
        if df.empty or len(df) < self.cfg.EMA_TREND + 10:
            return
        df = self.ind.populate(df, self.cfg)

        position = self.pm.get(symbol)

        # ── GESTIONE POSIZIONE APERTA ─────────────────────────────────────────
        if position:
            current_price = float(df.iloc[-1]["close"])

            # In livetest: controlla SL/TP sull'ultima candela chiusa (H/L)
            # prima di aggiornare il trailing stop, per fedeltà con il backtest
            if self.paper_trading:
                exit_triggered, reason, exit_price = self._livetest_check_sl_tp(
                    symbol, df, position)
                if exit_triggered:
                    pnl, _ = self.vp.simulate_sell(exit_price, position["qty"],
                                                   position["invested"])
                    self.pm.close_position(symbol, exit_price, position["qty"], reason)
                    self.tg.send(
                        f"📤 *[VT] SELL {symbol}*\n"
                        f"Prezzo uscita: {exit_price:.4f}\n"
                        f"PnL virtuale: {pnl:+.2f} USDC\n"
                        f"Motivo: {reason}\n"
                        f"Cash: {self.vp.balance:.2f} USDC")
                    return

            # Aggiorna trailing stop (identico per live e livetest)
            position = self.rm.update_trailing_stop(position, current_price)
            self.pm.set(symbol, position)

            # Controlla segnale di uscita da indicatori
            sell, reason = self.strat.check_sell(df, position)
            if sell:
                if self.paper_trading:
                    pnl, _ = self.vp.simulate_sell(current_price, position["qty"],
                                                   position["invested"])
                    self.pm.close_position(symbol, current_price, position["qty"], reason)
                    self.tg.send(
                        f"📤 *[VT] SELL {symbol}*\n"
                        f"Prezzo: {current_price:.4f}\n"
                        f"PnL virtuale: {pnl:+.2f} USDC\n"
                        f"Motivo: {reason}\n"
                        f"Cash: {self.vp.balance:.2f} USDC")
                else:
                    asset = symbol.replace(self.cfg.QUOTE_ASSET, "")
                    qty   = self.bc.get_asset_balance(asset)
                    if qty > 0:
                        order = self.bc.sell_market_qty(symbol, qty)
                        if order:
                            pnl = self.pm.close_position(
                                symbol, current_price, qty, reason)
                            self.tg.send(
                                f"📤 *SELL {symbol}*\n"
                                f"Prezzo: {current_price:.4f}\n"
                                f"PnL: {pnl:+.2f} USDC\n"
                                f"Motivo: {reason}")
            return

        # ── VALUTAZIONE INGRESSO ──────────────────────────────────────────────
        if halted or not self.tg.trading_enabled:
            return
        if self.pm.open_count() >= self.cfg.MAX_OPEN_POSITIONS:
            return

        ob_imb  = self.bc.ob_imbalance(symbol)
        signal  = self.strat.score_buy(df, ob_imb)
        last    = df.iloc[-1]

        self.log.debug(
            f"{symbol} score={signal.score:.1f} "
            f"(soglia={self.cfg.SIGNAL_THRESHOLD}) OBI={ob_imb:.2f}")

        if not signal.buy:
            return

        # Dimensione posizione (Kelly) sull'equity corrente
        current_equity = self.vp.balance if self.paper_trading else equity
        invest = self.rm.kelly_position_size(
            win_rate, current_equity, signal.atr, last["close"])
        if invest < self.cfg.MIN_USDC_TRADE:
            self.log.info(f"{symbol}: Kelly {invest:.2f} < minimo. Skip.")
            return
        invest = min(invest, current_equity * 0.95)

        if self.paper_trading:
            # ── Acquisto virtuale ─────────────────────────────────────────────
            if self.vp.balance < invest:
                self.log.info(
                    f"[VT] {symbol}: balance virtuale {self.vp.balance:.2f} < "
                    f"invest {invest:.2f}. Skip.")
                return

            qty, entry_price = self.vp.simulate_buy(last["close"], invest)
            self.pm.open_position(
                symbol      = symbol,
                entry_price = entry_price,
                qty         = qty,
                invested    = invest,
                sl          = signal.sl_price,
                tp          = signal.tp_price,
                atr         = signal.atr,
            )
            reasons_txt = "\n".join(f"  {r}" for r in signal.reasons)
            self.tg.send(
                f"📥 *[VT] BUY {symbol}*\n"
                f"Prezzo: {entry_price:.4f}\n"
                f"Qty: {qty:.6f}\n"
                f"Investito: {invest:.2f} USDC (virtuale)\n"
                f"Score: {signal.score:.1f}/100\n"
                f"SL: {signal.sl_price:.4f} | TP: {signal.tp_price:.4f}\n"
                f"Cash rimasto: {self.vp.balance:.2f} USDC\n\n"
                f"Segnali:\n{reasons_txt}")
            self.log.info(
                f"[VT] BUY {symbol} qty={qty:.6f} @ {entry_price:.4f} "
                f"score={signal.score:.1f} invest={invest:.2f}USDC")

        else:
            # ── Acquisto reale ────────────────────────────────────────────────
            ref_price = last["close"]
            order     = self.bc.buy_market_quote(symbol, invest, ref_price)
            if not order:
                return
            filled_qty   = float(order.get("executedQty", 0))
            executed_val = float(order.get("cummulativeQuoteQty", invest))
            entry_price  = executed_val / filled_qty if filled_qty > 0 else ref_price
            self.pm.open_position(
                symbol      = symbol,
                entry_price = entry_price,
                qty         = filled_qty,
                invested    = executed_val,
                sl          = signal.sl_price,
                tp          = signal.tp_price,
                atr         = signal.atr,
            )
            reasons_txt = "\n".join(f"  {r}" for r in signal.reasons)
            self.tg.send(
                f"📥 *BUY {symbol}*\n"
                f"Prezzo: {entry_price:.4f}\n"
                f"Qty: {filled_qty:.6f}\n"
                f"Investito: {executed_val:.2f} USDC\n"
                f"Score: {signal.score:.1f}/100\n"
                f"SL: {signal.sl_price:.4f} | TP: {signal.tp_price:.4f}\n\n"
                f"Segnali:\n{reasons_txt}")
            self.log.info(
                f"BUY {symbol} qty={filled_qty:.6f} @ {entry_price:.4f} "
                f"score={signal.score:.1f} kelly={invest:.2f}USDC")

    # ── Helper livetest ───────────────────────────────────────────────────────

    def _livetest_check_sl_tp(self, symbol: str, df: pd.DataFrame,
                               position: dict) -> Tuple[bool, str, float]:
        """
        Controlla SL e TP usando i valori high/low dell'ultima candela CHIUSA
        (penultima riga del DataFrame, dato che l'ultima è ancora in formazione).

        Priorità: SL > TP (approccio conservativo, come nel backtest).
        Ritorna (exit_triggered, reason, exit_price).
        """
        if len(df) < 2:
            return False, "", 0.0

        candle = df.iloc[-2]   # ultima candela chiusa
        sl     = position.get("trailing_sl", position.get("sl_price", 0))
        tp     = position.get("tp_price", 0)

        if sl and float(candle["low"]) <= sl:
            return True, f"🛑 SL colpito (low={candle['low']:.4f} ≤ {sl:.4f})", sl

        if tp and float(candle["high"]) >= tp:
            return True, f"🎯 TP colpito (high={candle['high']:.4f} ≥ {tp:.4f})", tp

        return False, "", 0.0

    def _livetest_report(self) -> str:
        """Genera un report testuale completo dello stato del livetest."""
        st      = self.pm.stats()
        prices  = {}
        for sym in self.pm.positions:
            if self.pm.positions[sym]:
                try:
                    prices[sym] = self.bc.get_ticker_price(sym)
                except Exception:
                    pass
        equity  = self.vp.mark_to_market(self.pm.positions, prices)
        ret_pct = (equity - self.vp.initial_capital) / self.vp.initial_capital * 100
        W = 55
        lines = [
            f"{'─'*W}",
            f"  📝 LIVETEST REPORT — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
            f"{'─'*W}",
            f"  Capitale iniziale : {self.vp.initial_capital:>10.2f} USDC",
            f"  Cash disponibile  : {self.vp.balance:>10.2f} USDC",
            f"  Equity (MTM)      : {equity:>10.2f} USDC",
            f"  Rendimento        : {ret_pct:>+10.2f}%",
            f"  Posizioni aperte  : {self.pm.open_count()}",
            f"{'─'*W}",
            f"  Trade chiuse      : {st['total']}",
            f"  Win rate          : {st['win_rate']*100:>10.1f}%",
            f"  PnL totale        : {st['total_pnl']:>+10.2f} USDC",
            f"  PnL medio         : {st.get('avg_pnl', 0):>+10.2f} USDC",
        ]
        if st["total"] > 0:
            lines += [
                f"  Miglior trade     : {st.get('best',  0):>+10.2f} USDC",
                f"  Peggior trade     : {st.get('worst', 0):>+10.2f} USDC",
            ]
        # Dettaglio posizioni aperte
        open_pos = {k: v for k, v in self.pm.positions.items() if v}
        if open_pos:
            lines.append(f"{'─'*W}")
            lines.append(f"  Posizioni aperte:")
            for sym, pos in open_pos.items():
                p   = prices.get(sym, pos["entry_price"])
                pnl = (p - pos["entry_price"]) * pos["qty"]
                pct = pnl / pos["invested"] * 100 if pos["invested"] else 0
                lines.append(
                    f"    {sym:<12} entry={pos['entry_price']:.4f} "
                    f"now={p:.4f} PnL={pnl:+.2f} ({pct:+.1f}%)")
        lines.append(f"{'─'*W}")
        return "\n".join(lines)

    def _livetest_telegram_summary(self) -> str:
        """Versione compatta per Telegram del report livetest."""
        st      = self.pm.stats()
        prices  = {}
        for sym in self.pm.positions:
            if self.pm.positions[sym]:
                try:
                    prices[sym] = self.bc.get_ticker_price(sym)
                except Exception:
                    pass
        equity  = self.vp.mark_to_market(self.pm.positions, prices)
        ret_pct = (equity - self.vp.initial_capital) / self.vp.initial_capital * 100
        emoji   = "🟢" if ret_pct >= 0 else "🔴"
        return (
            f"{emoji} Equity: {equity:.2f} USDC ({ret_pct:+.2f}%)\n"
            f"💵 Cash: {self.vp.balance:.2f} | 📈 Pos: {self.pm.open_count()}\n"
            f"📋 Trade: {st['total']} | WR: {st['win_rate']*100:.1f}% | "
            f"PnL: {st['total_pnl']:+.2f} USDC"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  11. BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """Backtest vettoriale su dati storici Binance."""

    COMMISSION = 0.001
    SLIPPAGE   = 0.0005

    def __init__(self, cfg: Config):
        self.cfg   = cfg
        self.log   = build_logger(cfg.LOG_FILE)
        self.strat = Strategy(cfg)
        self.ind   = IndicatorEngine()
        self.rm    = RiskManager(cfg, self.log)

    def _fetch_klines(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        end   = datetime.utcnow()
        start = end - timedelta(days=days)
        url   = "https://api.binance.com/api/v3/klines"
        all_rows = []
        current_start = start
        while True:
            params = {
                "symbol": symbol, "interval": interval, "limit": 1000,
                "startTime": int(current_start.timestamp() * 1000),
                "endTime":   int(end.timestamp() * 1000),
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                break
            rows = resp.json()
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = pd.Timestamp(rows[-1][0], unit="ms")
            current_start = last_ts + timedelta(milliseconds=1)
            if len(rows) < 1000 or last_ts >= end:
                break
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["open_time", "open", "high", "low", "close", "volume"]]

    def run(self, days: int = 500, initial_capital: float = 100.0):
        wallet     = initial_capital
        all_trades = []
        print(f"\n{'═'*60}")
        print(f"  BACKTEST — {days} giorni | Capitale: {initial_capital:.2f} USDC")
        print(f"{'═'*60}\n")

        for symbol in self.cfg.SYMBOLS:
            print(f"\n─── {symbol} ───")
            df_raw = self._fetch_klines(symbol, self.cfg.INTERVAL, days)
            if df_raw.empty:
                print(f"  Nessun dato per {symbol}")
                continue
            df = self.ind.populate(df_raw, self.cfg)
            print(f"  Candele: {len(df)} | dal {df['open_time'].iloc[0].date()}")

            position       = None
            trades         = []
            trade_log_local = []

            for i in range(self.cfg.EMA_TREND + 5, len(df)):
                window = df.iloc[:i + 1]
                row    = window.iloc[-1]
                price  = row["close"]
                date   = row["open_time"].strftime("%Y-%m-%d")

                if position:
                    new_sl = price - self.cfg.ATR_STOP_MULT * position["atr"]
                    if new_sl > position["trailing_sl"]:
                        position["trailing_sl"] = new_sl
                    sell, reason = self.strat.check_sell(window, position)
                    if sell:
                        exit_p = price * (1 - self.SLIPPAGE)
                        comm   = exit_p * position["qty"] * self.COMMISSION
                        pnl    = (exit_p - position["entry_price"]) * position["qty"] - comm
                        wallet += pnl
                        trades.append({
                            "symbol": symbol, "entry": position["entry_price"],
                            "exit": exit_p, "qty": position["qty"],
                            "pnl": pnl, "pnl_pct": pnl / position["invested"] * 100,
                            "reason": reason, "date": date,
                        })
                        trade_log_local.append({"pnl": pnl})
                        emoji = "🟢" if pnl > 0 else "🔴"
                        print(f"  {emoji} {date} SELL @ {exit_p:.4f} | "
                              f"PnL={pnl:+.2f} | {reason}")
                        position = None
                    continue

                signal  = self.strat.score_buy(window, ob_imbalance=0.5)
                if not signal.buy:
                    continue
                wr      = self.rm.estimate_win_rate(trade_log_local)
                invest  = self.rm.kelly_position_size(wr, wallet, signal.atr, price)
                invest  = min(invest, wallet * 0.95)
                if invest < self.cfg.MIN_USDC_TRADE:
                    continue
                entry_p = price * (1 + self.SLIPPAGE)
                comm    = invest * self.COMMISSION
                qty     = (invest - comm) / entry_p
                wallet -= invest
                position = {
                    "entry_price":  entry_p,
                    "qty":          qty,
                    "invested":     invest,
                    "sl_price":     signal.sl_price,
                    "trailing_sl":  signal.sl_price,
                    "tp_price":     signal.tp_price,
                    "atr":          signal.atr,
                }
                print(f"  📈 {date} BUY  @ {entry_p:.4f} | "
                      f"Score={signal.score:.0f} Kelly={invest:.2f}USDC")
            all_trades.extend(trades)
            if trades:
                pnls    = [t["pnl"] for t in trades]
                wins    = sum(1 for p in pnls if p > 0)
                tot_pnl = sum(pnls)
                print(f"\n  Trade: {len(trades)} | Wins: {wins} "
                      f"({wins/len(trades)*100:.0f}%) | PnL: {tot_pnl:+.2f} USDC")

        print(f"\n{'═'*60}")
        if all_trades:
            pnls = [t["pnl"] for t in all_trades]
            wins = sum(1 for p in pnls if p > 0)
            print(f"  RIEPILOGO GLOBALE")
            print(f"  Trade totali  : {len(pnls)}")
            print(f"  Win rate      : {wins/len(pnls)*100:.1f}%")
            print(f"  PnL totale    : {sum(pnls):+.2f} USDC")
            print(f"  Miglior trade : {max(pnls):+.2f} USDC")
            print(f"  Peggior trade : {min(pnls):+.2f} USDC")
        print(f"  Capitale finale: {wallet:.2f} USDC "
              f"(rendimento: {(wallet/initial_capital-1)*100:+.2f}%)")
        print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Paper trading live con 2000 USDC virtuali
  python bot_v2.py --livetest --capital 2000

  # Livetest con simboli e intervallo personalizzati
  python bot_v2.py --livetest --capital 1000 --symbols BTCUSDC,ETHUSDC --interval 4h

  # Trading reale (richiede API_KEY e API_SECRET in Config)
  python bot_v2.py --live

  # Backtest su 1 anno di storico
  python bot_v2.py --backtest --days 365 --capital 500
        """)

    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--live",     action="store_true",
                          help="Trading live reale su Binance (richiede API key)")
    mode_grp.add_argument("--livetest", action="store_true",
                          help="Paper trading live: portafoglio virtuale, dati reali")
    mode_grp.add_argument("--backtest", action="store_true",
                          help="Backtest su dati storici")

    parser.add_argument("--days",      type=int,   default=500,
                        help="Giorni per il backtest (default: 500)")
    parser.add_argument("--capital",   type=float, default=1000.0,
                        help="Capitale iniziale USDC — livetest o backtest (default: 1000)")
    parser.add_argument("--symbols",   type=str, default=None,
                        help="Simboli separati da virgola (es. BTCUSDC,ETHUSDC)")
    parser.add_argument("--interval",  type=str, default=None,
                        help="Intervallo candele (es. 1h, 4h, 1d)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Soglia score segnale (default 60)")

    args = parser.parse_args()

    # ── Config — INSERISCI LE TUE CREDENZIALI QUI ─────────────────────────────
    cfg = Config(
        # API_KEY    = "la_tua_chiave",
        # API_SECRET = "il_tuo_secret",
        # TELEGRAM_TOKEN   = "...",
        # TELEGRAM_CHAT_ID = "...",
    )

    # Override da riga di comando
    if args.symbols:
        cfg.SYMBOLS = [s.strip().upper() for s in args.symbols.split(",")]
    if args.interval:
        cfg.INTERVAL = args.interval
    if args.threshold is not None:
        cfg.SIGNAL_THRESHOLD = args.threshold

    # ── Modalità ──────────────────────────────────────────────────────────────
    if args.live:
        if not cfg.API_KEY or not cfg.API_SECRET:
            print("ERRORE: --live richiede API_KEY e API_SECRET in Config.")
            sys.exit(1)
        bot = TradingBot(cfg, paper_trading=False)
        bot.start()

    elif args.livetest:
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║          📝  LIVETEST — PAPER TRADING LIVE                  ║
╠══════════════════════════════════════════════════════════════╣
║  Capitale virtuale : {args.capital:<37.2f} ║
║  Simboli  : {', '.join(cfg.SYMBOLS):<47} ║
║  Intervallo: {cfg.INTERVAL:<47} ║
║  Soglia score: {cfg.SIGNAL_THRESHOLD:<44.1f} ║
╠══════════════════════════════════════════════════════════════╣
║  Nessun ordine reale verrà piazzato.                        ║
║  Dati di mercato: REALI (Binance REST API)                  ║
║  Commissioni simulate: 0.1% + slippage 0.02% per lato      ║
║  Stato salvato in: state_livetest.json / vp_livetest.json   ║
╚══════════════════════════════════════════════════════════════╝""")
        bot = TradingBot(cfg, paper_trading=True, paper_capital=args.capital)
        bot.start()

    elif args.backtest:
        bt = Backtester(cfg)
        bt.run(days=args.days, initial_capital=args.capital)
