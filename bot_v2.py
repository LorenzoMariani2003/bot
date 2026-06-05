"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CRYPTO TRADING BOT v2.0 — QUANT ARCHITECTURE                  ║
║                                                                              ║
║  Modules (in-file classes):                                                 ║
║    Config          → parametri globali e costanti                           ║
║    Logger          → logging su file + console con rotazione                ║
║    IndicatorEngine → calcolo vettoriale indicatori (EMA, ADX, RSI, ATR, BB)║
║    Strategy        → logica multi-segnale con scoring pesato                ║
║    RiskManager     → Kelly frazionato, trailing stop ATR, daily drawdown    ║
║    BinanceClient   → wrapper async con retry esponenziale + jitter          ║
║    DataFeed        → WebSocket stream (ticker + depth) via asyncio          ║
║    PositionManager → stato posizioni, P&L, log CSV                         ║
║    TelegramHandler → comandi bot Telegram in thread separato                ║
║    TradingBot      → orchestratore principale (event loop asyncio)          ║
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
    INTERVAL:   str = "1h"        # candle interval per strategia principale
    CANDLE_LIMIT: int = 300       # storico candele da caricare all'avvio

    # ── Gestione del rischio ─────────────────────────────────────────────────
    MAX_OPEN_POSITIONS: int   = 5
    KELLY_FRACTION:     float = 0.25   # Quarter-Kelly: riduce varianza
    MIN_KELLY_BET:      float = 0.02   # 2% minimo portafoglio per trade
    MAX_KELLY_BET:      float = 0.20   # 20% massimo portafoglio per trade
    MIN_USDC_TRADE:     float = 11.0   # importo minimo per ordine (Binance: 10)
    DAILY_DRAWDOWN_LIMIT: float = 0.05 # -5%: blocco automatico del trading

    # ── ATR Trailing Stop ────────────────────────────────────────────────────
    ATR_PERIOD:      int   = 14
    ATR_STOP_MULT:   float = 2.0   # Stop Loss = entry - 2 * ATR
    ATR_TP_MULT:     float = 3.0   # Take Profit = entry + 3 * ATR (R:R = 1.5)

    # ── Parametri indicatori ─────────────────────────────────────────────────
    EMA_FAST:    int = 16
    EMA_SLOW:    int = 21
    EMA_TREND:   int = 200
    ADX_PERIOD:  int = 10
    ADX_MIN:     float = 22.0     # sotto questo valore il trend è debole
    RSI_PERIOD:  int = 14
    RSI_OB:      float = 70.0     # overbought (evita entrate long)
    RSI_OS:      float = 30.0     # oversold (conferma entrate long)
    BB_PERIOD:   int = 20
    BB_STD:      float = 2.0

    # ── Scoring segnali (pesi sommati → soglia entrata) ──────────────────────
    # Se score_buy >= SIGNAL_THRESHOLD → ordine BUY
    SIGNAL_THRESHOLD: float = 60.0   # su 100
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "ema_trend":      25.0,   # prezzo sopra EMA200
        "ema_cross":      20.0,   # EMA fast > EMA slow (crossover)
        "adx_strength":   15.0,   # ADX > soglia (trend forte)
        "rsi_ok":         15.0,   # RSI in zona favorevole (non OB)
        "bb_position":    10.0,   # prezzo vicino banda inferiore BB
        "volume_spike":   10.0,   # volume > 1.5x media 20 periodi
        "ob_imbalance":    5.0,   # Order Book: ask_vol < bid_vol
    })

    # ── Order Book ───────────────────────────────────────────────────────────
    OB_DEPTH_LEVELS: int   = 10      # livelli di profondità da analizzare
    OB_IMBALANCE_TH: float = 0.55    # bid/(bid+ask) > 55% → pressione rialzista

    # ── Slippage control ─────────────────────────────────────────────────────
    MAX_SLIPPAGE_PCT: float = 0.003  # rifiuta ordine se prezzo si muove >0.3%

    # ── Loop timing ──────────────────────────────────────────────────────────
    LOOP_SLEEP_SEC:   int = 60      # pausa tra un ciclo e l'altro
    WS_RECONNECT_SEC: int = 5       # attesa prima di riconnettere WebSocket

    # ── File paths ───────────────────────────────────────────────────────────
    LOG_FILE:    str = "bot_v2.log"
    TRADE_LOG:   str = "trades_v2.csv"
    STATE_FILE:  str = "state_v2.json"
    STATUS_FILE: str = "status_v2.json"


# ══════════════════════════════════════════════════════════════════════════════
#  2. LOGGER
# ══════════════════════════════════════════════════════════════════════════════

def build_logger(log_file: str, level=logging.INFO) -> logging.Logger:
    """Logger con rotazione file (max 5 MB × 3 backup). Thread-safe."""
    logger = logging.getLogger("TradingBot")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    # File con rotazione
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3,
                             encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
#  3. INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """
    Calcolo vettoriale puro con NumPy/Pandas.
    Tutti i metodi operano su DataFrame e restituiscono colonne aggiuntive.
    """

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int) -> pd.Series:
        """RSI classico di Wilder (usa EWM con alpha=1/period)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
        """
        Average True Range:
        TR = max(H-L, |H-Cp|, |L-Cp|)  dove Cp = close precedente
        ATR = EWM(TR, alpha=1/period)
        """
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        ADX, +DI, -DI secondo il metodo Wilder.
        ADX misura la FORZA del trend (non la direzione).
        ADX > 25 → trend forte; ADX < 20 → mercato laterale.
        """
        high, low, close = df["high"], df["low"], df["close"]
        prev_high = high.shift(1)
        prev_low  = low.shift(1)

        dm_plus  = (high - prev_high).clip(lower=0)
        dm_minus = (prev_low - low).clip(lower=0)
        # Annulla quando entrambe positive e la differenza non è dominante
        mask = dm_plus < dm_minus
        dm_plus[mask] = 0
        mask2 = dm_minus < dm_plus
        dm_minus[mask2] = 0

        atr_val = IndicatorEngine.atr(df, period)
        di_plus  = 100 * dm_plus.ewm(alpha=1/period, adjust=False).mean() / atr_val
        di_minus = 100 * dm_minus.ewm(alpha=1/period, adjust=False).mean() / atr_val
        dx = (100 * (di_plus - di_minus).abs() /
              (di_plus + di_minus).replace(0, np.nan))
        adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx_val, di_plus, di_minus

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int,
                        std_mult: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """BB: middle = SMA(period), upper/lower = middle ± std_mult*σ"""
        middle = series.rolling(period).mean()
        std    = series.rolling(period).std()
        return middle + std_mult * std, middle, middle - std_mult * std

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26,
             signal=9) -> Tuple[pd.Series, pd.Series]:
        ema_f = series.ewm(span=fast, adjust=False).mean()
        ema_s = series.ewm(span=slow, adjust=False).mean()
        macd_line   = ema_f - ema_s
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    def populate(self, df: pd.DataFrame, cfg: "Config") -> pd.DataFrame:
        """Applica tutti gli indicatori al DataFrame e li aggiunge come colonne."""
        df = df.copy()
        close = df["close"]

        df["ema_fast"]  = self.ema(close, cfg.EMA_FAST)
        df["ema_slow"]  = self.ema(close, cfg.EMA_SLOW)
        df["ema_trend"] = self.ema(close, cfg.EMA_TREND)
        df["rsi"]       = self.rsi(close, cfg.RSI_PERIOD)
        df["atr"]       = self.atr(df, cfg.ATR_PERIOD)

        adx_val, di_plus, di_minus = self.adx(df, cfg.ADX_PERIOD)
        df["adx"]       = adx_val
        df["di_plus"]   = di_plus
        df["di_minus"]  = di_minus

        bb_up, bb_mid, bb_low = self.bollinger_bands(
            close, cfg.BB_PERIOD, cfg.BB_STD)
        df["bb_upper"]  = bb_up
        df["bb_middle"] = bb_mid
        df["bb_lower"]  = bb_low

        macd_line, signal_line = self.macd(close)
        df["macd"]        = macd_line
        df["macd_signal"] = signal_line

        # Volume spike: True se volume > 1.5× media mobile 20 periodi
        vol_ma = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] > (vol_ma * 1.5)

        return df


# ══════════════════════════════════════════════════════════════════════════════
#  4. STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    score:    float          = 0.0
    buy:      bool           = False
    sell:     bool           = False
    reasons:  List[str]      = field(default_factory=list)
    atr:      float          = 0.0
    sl_price: float          = 0.0
    tp_price: float          = 0.0


class Strategy:
    """
    Strategia multi-segnale con scoring pesato.

    LOGICA DI ENTRATA (LONG):
    ─────────────────────────
    Ogni condizione contribuisce con un peso al punteggio totale (0-100).
    Se score ≥ SIGNAL_THRESHOLD → segnale BUY.

    Condizioni valutate:
      1. EMA Trend  : close > EMA200          (trend long-term rialzista)
      2. EMA Cross  : EMA_fast > EMA_slow     (trend medium-term rialzista)
      3. ADX        : ADX > ADX_MIN           (trend abbastanza forte)
      4. RSI        : RSI < RSI_OB            (non in zona ipercomprato)
      5. Bollinger  : close < BB_middle       (prezzo nella metà inferiore)
      6. Volume     : volume spike rilevato   (momentum confermato da volume)
      7. OB         : bid_vol > ask_vol (OB imbalance > soglia)

    LOGICA DI USCITA (SELL):
    ──────────────────────────
      - Trailing Stop Loss raggiunto (aggiornato con ATR)
      - Take Profit raggiunto
      - EMA cross ribassista: EMA_fast < EMA_slow (dopo essere stato >)
      - ADX crolla sotto soglia (trend finito)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ind = IndicatorEngine()

    def score_buy(self, df: pd.DataFrame,
                  ob_imbalance: float = 0.5) -> SignalResult:
        """
        Calcola il punteggio di acquisto basato sugli indicatori.
        ob_imbalance = bid_volume / (bid_volume + ask_volume)
        Valori > OB_IMBALANCE_TH indicano pressione rialzista.
        """
        cfg = self.cfg
        result = SignalResult()
        if len(df) < cfg.EMA_TREND + 5:
            result.reasons.append("Dati insufficienti per il calcolo")
            return result

        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.0
        w = cfg.WEIGHTS

        # 1. EMA Trend (close > EMA200)
        if last["close"] > last["ema_trend"]:
            score += w["ema_trend"]
            pct = (last["close"] / last["ema_trend"] - 1) * 100
            result.reasons.append(f"✓ EMA trend: prezzo +{pct:.2f}% sopra EMA{cfg.EMA_TREND}")
        else:
            result.reasons.append(f"✗ EMA trend: prezzo sotto EMA{cfg.EMA_TREND}")

        # 2. EMA Cross (fast > slow)
        if last["ema_fast"] > last["ema_slow"]:
            score += w["ema_cross"]
            result.reasons.append(f"✓ EMA cross: EMA{cfg.EMA_FAST} > EMA{cfg.EMA_SLOW}")
        else:
            result.reasons.append(f"✗ EMA cross: EMA{cfg.EMA_FAST} < EMA{cfg.EMA_SLOW}")

        # 3. ADX (forza del trend)
        if last["adx"] > cfg.ADX_MIN and last["di_plus"] > last["di_minus"]:
            score += w["adx_strength"]
            result.reasons.append(f"✓ ADX={last['adx']:.1f} (trend forte, +DI > -DI)")
        else:
            result.reasons.append(f"✗ ADX={last['adx']:.1f} (trend debole o laterale)")

        # 4. RSI (non in overbought)
        rsi_val = last["rsi"]
        if rsi_val < cfg.RSI_OB:
            partial = w["rsi_ok"] * (1 - max(0, rsi_val - cfg.RSI_OS) /
                                     (cfg.RSI_OB - cfg.RSI_OS))
            score += partial
            result.reasons.append(f"✓ RSI={rsi_val:.1f} (zona neutra/OS)")
        else:
            result.reasons.append(f"✗ RSI={rsi_val:.1f} (overbought, evita entrata)")

        # 5. Bollinger Bands (prezzo nella metà inferiore)
        bb_pos = (last["close"] - last["bb_lower"]) / (
                  last["bb_upper"] - last["bb_lower"] + 1e-9)
        if bb_pos < 0.5:
            score += w["bb_position"] * (1 - bb_pos * 2)
            result.reasons.append(f"✓ BB: prezzo a {bb_pos*100:.0f}% della banda ({bb_pos:.2f})")
        else:
            result.reasons.append(f"✗ BB: prezzo a {bb_pos*100:.0f}% della banda")

        # 6. Volume spike
        if last["vol_spike"]:
            score += w["volume_spike"]
            result.reasons.append("✓ Volume spike rilevato (> 1.5× media)")
        else:
            result.reasons.append("✗ Volume nella norma")

        # 7. Order Book Imbalance
        if ob_imbalance > cfg.OB_IMBALANCE_TH:
            score += w["ob_imbalance"]
            result.reasons.append(
                f"✓ OB Imbalance={ob_imbalance:.2f} (pressione rialzista)")
        else:
            result.reasons.append(f"✗ OB Imbalance={ob_imbalance:.2f}")

        # Calcola SL/TP basati su ATR
        atr_val = last["atr"]
        result.atr      = atr_val
        result.sl_price = last["close"] - cfg.ATR_STOP_MULT * atr_val
        result.tp_price = last["close"] + cfg.ATR_TP_MULT  * atr_val
        result.score    = round(score, 2)
        result.buy      = score >= cfg.SIGNAL_THRESHOLD
        return result

    def check_sell(self, df: pd.DataFrame, position: dict) -> Tuple[bool, str]:
        """
        Controlla se uscire dalla posizione.
        Restituisce (sell: bool, motivo: str).
        """
        if len(df) < 3:
            return False, ""
        last = df.iloc[-1]
        price = last["close"]

        # 1. Trailing Stop Loss (aggiornato da RiskManager)
        sl = position.get("trailing_sl", position.get("sl_price", 0))
        if sl and price <= sl:
            return True, f"🛑 Trailing Stop Loss: {price:.4f} ≤ {sl:.4f}"

        # 2. Take Profit fisso
        tp = position.get("tp_price", 0)
        if tp and price >= tp:
            return True, f"🎯 Take Profit: {price:.4f} ≥ {tp:.4f}"

        # 3. EMA cross ribassista
        if last["ema_fast"] < last["ema_slow"]:
            # Conferma con ADX in calo: evita falsi segnali
            if last["adx"] < self.cfg.ADX_MIN or last["di_minus"] > last["di_plus"]:
                return True, "📉 EMA cross ribassista + ADX debole/inversione"

        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
#  5. RISK MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Gestisce la dimensione della posizione e i limiti di rischio.

    KELLY CRITERION (frazionato):
    ─────────────────────────────
    f* = (W * R - L) / R
    dove:
      W = win rate storica  (es. 0.55)
      L = 1 - W            (loss rate)
      R = reward/risk ratio (es. ATR_TP_MULT / ATR_STOP_MULT = 1.5)

    Si usa f* × KELLY_FRACTION (quarter-Kelly di default) per ridurre
    la varianza e proteggersi da stime imprecise di W.
    Il risultato viene clampato in [MIN_KELLY_BET, MAX_KELLY_BET].
    """

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg    = cfg
        self.log    = logger
        self._daily_start_equity: Optional[float] = None
        self._daily_date:         Optional[str]   = None
        self.trading_halted:      bool             = False

    def kelly_position_size(self, win_rate: float,
                             equity: float,
                             atr: float,
                             entry_price: float) -> float:
        """
        Ritorna l'importo in USDC da investire per questa trade.

        Parametri:
          win_rate   : win rate stimata [0, 1]
          equity     : USDC disponibile
          atr        : ATR corrente (in USDC)
          entry_price: prezzo di entrata
        """
        cfg = self.cfg
        # Ratio R = reward/risk basato su multipli ATR
        R = cfg.ATR_TP_MULT / cfg.ATR_STOP_MULT   # default: 3/2 = 1.5
        L = 1 - win_rate

        # f* = (W*R - L) / R  (Kelly formula)
        numerator = win_rate * R - L
        if numerator <= 0:
            self.log.warning(
                f"Kelly negativo (W={win_rate:.2f}, R={R:.2f}): edge negativo, skip")
            return 0.0

        f_star   = numerator / R
        f_scaled = f_star * cfg.KELLY_FRACTION   # Quarter-Kelly

        # Clampa tra min e max
        f_final = max(cfg.MIN_KELLY_BET, min(cfg.MAX_KELLY_BET, f_scaled))
        amount  = equity * f_final

        self.log.info(
            f"Kelly: W={win_rate:.2f} R={R:.2f} f*={f_star:.3f} "
            f"f_scaled={f_scaled:.3f} f_final={f_final:.3f} → {amount:.2f} USDC")
        return amount

    def update_trailing_stop(self, position: dict, current_price: float) -> dict:
        """
        Aggiorna il trailing stop se il prezzo è salito.
        Trailing SL = max(prezzo corrente - ATR_STOP_MULT × ATR, SL precedente)
        """
        atr        = position.get("atr", 0)
        new_sl     = current_price - self.cfg.ATR_STOP_MULT * atr
        current_sl = position.get("trailing_sl", position.get("sl_price", 0))
        if new_sl > current_sl:
            position["trailing_sl"] = new_sl
        return position

    def check_daily_drawdown(self, current_equity: float) -> bool:
        """
        Controlla se la perdita giornaliera supera il limite.
        Ritorna True se il trading deve essere bloccato.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if self._daily_date != today:
            self._daily_date         = today
            self._daily_start_equity = current_equity
            self.trading_halted      = False
            self.log.info(f"Nuovo giorno: equity di partenza = {current_equity:.2f} USDC")

        if self._daily_start_equity and self._daily_start_equity > 0:
            dd = (self._daily_start_equity - current_equity) / self._daily_start_equity
            if dd >= self.cfg.DAILY_DRAWDOWN_LIMIT:
                if not self.trading_halted:
                    self.trading_halted = True
                    self.log.error(
                        f"⚠️ DAILY DRAWDOWN LIMIT raggiunto: -{dd*100:.2f}% "
                        f"(limite: -{self.cfg.DAILY_DRAWDOWN_LIMIT*100:.1f}%). "
                        f"Trading BLOCCATO per oggi.")
                return True
        return False

    def estimate_win_rate(self, trade_log: List[dict]) -> float:
        """
        Stima la win rate dalle ultime N trade chiuse.
        Usa le ultime 50 trade per relevanza statistica.
        Se non ci sono dati sufficienti restituisce 0.50 (neutro).
        """
        closed = [t for t in trade_log if t.get("pnl") is not None][-50:]
        if len(closed) < 5:
            return 0.50
        wins = sum(1 for t in closed if t["pnl"] > 0)
        return wins / len(closed)


# ══════════════════════════════════════════════════════════════════════════════
#  6. BINANCE CLIENT (SYNC WRAPPER CON RETRY)
# ══════════════════════════════════════════════════════════════════════════════

class BinanceClient:
    """
    Wrapper attorno al Client sincrono di python-binance con:
    - Retry esponenziale con jitter per errori transitori
    - Rate limiting (contatore interno delle chiamate al minuto)
    - Controllo slippage prima dell'esecuzione dell'ordine
    """

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg    = cfg
        self.log    = logger
        self.client: Optional[Client] = None
        self._request_times: deque = deque(maxlen=1200)  # ultime 1200 req

    def connect(self):
        if not _HAS_BINANCE:
            raise RuntimeError("python-binance non installato")
        self.client = Client(self.cfg.API_KEY, self.cfg.API_SECRET)
        self.client.session.timeout = 30
        self.log.info("Connessione Binance stabilita")

    def _rate_check(self):
        """Rispetta il limite di 1200 richieste/minuto di Binance."""
        now = time.time()
        # Rimuovi richieste più vecchie di 60s
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        if len(self._request_times) >= 1100:   # margine di sicurezza
            sleep_time = 60 - (now - self._request_times[0]) + 1
            self.log.warning(f"Rate limit: attendo {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._request_times.append(time.time())

    def request(self, func, *args, max_retries=4, **kwargs):
        """
        Esegue una chiamata API con retry esponenziale + jitter.
        Jitter evita la sincronizzazione di più bot sullo stesso server.
        """
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
                    self.log.warning(
                        f"[{func.__name__}] tentativo {attempt+1} fallito: {e}. "
                        f"Retry in {wait:.1f}s")
                    time.sleep(wait)
                    delay *= 2          # backoff esponenziale
                else:
                    self.log.error(f"[{func.__name__}] tutti i retry falliti: {e}")
                    raise
            except BinanceAPIException as e:
                if e.code in (-1003, -1015):   # too many requests
                    time.sleep(60)
                    continue
                raise
        return None

    def get_klines(self, symbol: str, interval: str,
                   limit: int = 300) -> pd.DataFrame:
        """Scarica le candele OHLCV e le restituisce come DataFrame."""
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
        """Restituisce order book grezzo."""
        return self.request(self.client.get_order_book,
                            symbol=symbol, limit=limit) or {}

    def ob_imbalance(self, symbol: str) -> float:
        """
        Calcola Order Book Imbalance:
          imbalance = Σ bid_qty / (Σ bid_qty + Σ ask_qty)
          > 0.5 → più pressione rialzista (bid)
          < 0.5 → più pressione ribassista (ask)
        Considera solo i top N livelli (OB_DEPTH_LEVELS).
        """
        book = self.get_order_book(symbol, limit=self.cfg.OB_DEPTH_LEVELS * 2)
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
        """
        Ordine BUY a mercato con controllo slippage.
        Se il prezzo corrente si discosta più di MAX_SLIPPAGE_PCT
        dal prezzo di riferimento, l'ordine viene annullato.
        """
        current = self.get_ticker_price(symbol)
        if ref_price > 0:
            slip = abs(current - ref_price) / ref_price
            if slip > self.cfg.MAX_SLIPPAGE_PCT:
                self.log.warning(
                    f"Slippage eccessivo {symbol}: {slip*100:.3f}% > "
                    f"{self.cfg.MAX_SLIPPAGE_PCT*100:.3f}%. Ordine annullato.")
                return None

        q = float(Decimal(str(quote_amount)).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN))
        if q < self.cfg.MIN_USDC_TRADE:
            self.log.warning(f"Quote amount {q} < minimo {self.cfg.MIN_USDC_TRADE}")
            return None

        return self.request(
            self.client.create_order,
            symbol=symbol, side="BUY", type="MARKET", quoteOrderQty=q)

    def sell_market_qty(self, symbol: str, qty: float) -> Optional[dict]:
        """Ordine SELL a mercato per tutta la quantità disponibile."""
        info = self.request(self.client.get_symbol_info, symbol)
        if not info:
            return None
        # Adatta qty allo stepSize
        step = next((float(f["stepSize"])
                     for f in info["filters"] if f["filterType"] == "LOT_SIZE"),
                    0.001)
        decimals = max(0, -int(math.floor(math.log10(step))))
        adj_qty  = float(
            (Decimal(str(qty)) / Decimal(str(step))).to_integral_value(
                ROUND_DOWN) * Decimal(str(step)))
        if adj_qty <= 0:
            return None
        return self.request(
            self.client.create_order,
            symbol=symbol, side="SELL", type="MARKET",
            quantity=round(adj_qty, decimals))


# ══════════════════════════════════════════════════════════════════════════════
#  7. POSITION MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class PositionManager:
    """Mantiene lo stato delle posizioni aperte con persistenza JSON e log CSV."""

    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg       = cfg
        self.log       = logger
        self.positions: Dict[str, Optional[dict]] = {}
        self.trade_log: List[dict] = []

    def load(self):
        if os.path.exists(self.cfg.STATE_FILE):
            try:
                with open(self.cfg.STATE_FILE) as f:
                    state = json.load(f)
                self.positions = state.get("positions", {})
                self.log.info(f"Stato caricato da {self.cfg.STATE_FILE}")
            except Exception as e:
                self.log.error(f"Errore caricamento stato: {e}")
        # Carica trade log
        if os.path.exists(self.cfg.TRADE_LOG):
            try:
                with open(self.cfg.TRADE_LOG, newline="") as f:
                    self.trade_log = list(csv.DictReader(f))
            except Exception:
                pass

    def save(self):
        try:
            with open(self.cfg.STATE_FILE, "w") as f:
                json.dump({"ts": datetime.utcnow().isoformat(),
                           "positions": self.positions}, f, indent=2)
        except Exception as e:
            self.log.error(f"Errore salvataggio stato: {e}")

    def open_position(self, symbol: str, entry_price: float,
                      qty: float, invested: float,
                      sl: float, tp: float, atr: float):
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

    def close_position(self, symbol: str, exit_price: float, qty: float, reason: str):
        pos = self.positions.get(symbol)
        pnl = 0.0
        if pos:
            pnl = (exit_price - pos["entry_price"]) * qty
            pnl_pct = pnl / pos["invested"] * 100 if pos["invested"] else 0
            self._csv_log(symbol, "SELL", exit_price, qty, pos["invested"], pnl, pnl_pct)
            self.trade_log.append({
                "symbol": symbol, "pnl": pnl,
                "pnl_pct": pnl_pct, "reason": reason
            })
            self.log.info(
                f"CLOSE {symbol} @ {exit_price:.4f} | "
                f"PnL={pnl:+.2f} USDC ({pnl_pct:+.2f}%) | {reason}")
        self.positions[symbol] = None
        self.save()
        return pnl

    def _csv_log(self, symbol, action, price, qty, invested,
                 pnl=None, pnl_pct=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol, "action": action,
            "price": price, "qty": qty, "invested": invested,
            "pnl": pnl or 0, "pnl_pct": pnl_pct or 0,
        }
        exists = os.path.exists(self.cfg.TRADE_LOG)
        try:
            with open(self.cfg.TRADE_LOG, "a", newline="") as f:
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
        closed = [t for t in self.trade_log if "pnl" in t]
        if not closed:
            return {"total": 0, "wins": 0, "total_pnl": 0,
                    "win_rate": 0.5, "avg_pnl": 0}
        pnls  = [float(t["pnl"]) for t in closed]
        wins  = sum(1 for p in pnls if p > 0)
        return {
            "total":    len(pnls),
            "wins":     wins,
            "total_pnl": sum(pnls),
            "win_rate": wins / len(pnls),
            "avg_pnl":  sum(pnls) / len(pnls),
            "best":     max(pnls),
            "worst":    min(pnls),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  8. TELEGRAM HANDLER
# ══════════════════════════════════════════════════════════════════════════════

class TelegramHandler:
    """Gestisce i comandi Telegram in un thread separato."""

    def __init__(self, cfg: Config, logger: logging.Logger,
                 pm: PositionManager, bc: BinanceClient):
        self.cfg    = cfg
        self.log    = logger
        self.pm     = pm
        self.bc     = bc
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
            equity = self.bc.get_usdc_balance()
            op     = self.pm.open_count()
            ts     = "🟢 ATTIVO" if self.trading_enabled else "🔴 SOSPESO"
            return (
                f"🤖 *BOT v2.0 STATUS*\n\n"
                f"⏰ Uptime: {self._uptime()}\n"
                f"🎛️ Trading: {ts}\n"
                f"💰 USDC libero: {equity:.2f}\n"
                f"📈 Posizioni aperte: {op}/{self.cfg.MAX_OPEN_POSITIONS}\n\n"
                f"📋 Trade chiuse: {st['total']}\n"
                f"🎯 Win rate: {st['win_rate']*100:.1f}%\n"
                f"💎 PnL totale: {st['total_pnl']:+.2f} USDC\n"
                f"📊 PnL medio: {st.get('avg_pnl',0):+.2f} USDC\n"
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
            lines = ["📈 *POSIZIONI APERTE*\n"]
            for sym, pos in open_pos.items():
                price = self.bc.get_ticker_price(sym)
                pnl   = (price - pos["entry_price"]) * pos["qty"]
                pct   = pnl / pos["invested"] * 100 if pos["invested"] else 0
                lines.append(
                    f"*{sym}*: entry={pos['entry_price']:.4f} "
                    f"now={price:.4f} PnL={pnl:+.2f}USDC ({pct:+.1f}%)\n"
                    f"  SL={pos.get('trailing_sl',0):.4f} "
                    f"TP={pos.get('tp_price',0):.4f}")
            return "\n".join(lines)

        if low in ("/help", "help"):
            return (
                "🤖 *COMANDI BOT v2.0*\n\n"
                "/status — stato completo\n"
                "/positions — posizioni aperte + SL/TP\n"
                "/stop — sospendi nuovi acquisti\n"
                "/start — riattiva acquisti\n"
                "/help — questo messaggio"
            )

        return "❓ Comando non riconosciuto. Usa /help."

    def run_forever(self):
        self.log.info("Telegram handler avviato")
        while True:
            try:
                updates = self._get_updates()
                for upd in updates:
                    self._last_update_id = upd.get("update_id", 0)
                    msg = upd.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id == self.cfg.TELEGRAM_CHAT_ID and "text" in msg:
                        text = msg["text"]
                        self.log.info(f"Telegram cmd: {text}")
                        reply = self.handle(text)
                        self.send(reply)
            except Exception as e:
                self.log.error(f"Telegram error: {e}")
            time.sleep(2)


# ══════════════════════════════════════════════════════════════════════════════
#  9. TRADING BOT (ORCHESTRATORE)
# ══════════════════════════════════════════════════════════════════════════════

class TradingBot:
    """
    Orchestratore principale. Gira in un loop sincrono ogni LOOP_SLEEP_SEC.
    Per ogni simbolo:
      1. Scarica le candele più recenti
      2. Calcola gli indicatori
      3. Calcola OB Imbalance (real-time via REST)
      4. Valuta segnali di entrata/uscita
      5. Esegue gli ordini con controllo del rischio
    """

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.log  = build_logger(cfg.LOG_FILE)
        self.bc   = BinanceClient(cfg, self.log)
        self.pm   = PositionManager(cfg, self.log)
        self.strat = Strategy(cfg)
        self.ind   = IndicatorEngine()
        self.rm    = RiskManager(cfg, self.log)
        self.tg    = TelegramHandler(cfg, self.log, self.pm, self.bc)
        self._candle_cache: Dict[str, pd.DataFrame] = {}

    def start(self):
        self.bc.connect()
        self.pm.load()

        # Inizializza posizioni per simboli mancanti
        for sym in self.cfg.SYMBOLS:
            if sym not in self.pm.positions:
                self.pm.positions[sym] = None

        self.log.info(
            f"Bot avviato su {self.cfg.SYMBOLS} [{self.cfg.INTERVAL}]")
        self.tg.send(
            f"🤖 *Bot v2.0 avviato*\n"
            f"Simboli: {', '.join(self.cfg.SYMBOLS)}\n"
            f"Intervallo: {self.cfg.INTERVAL}\n"
            f"Usa /status per monitorare.")

        # Thread Telegram separato (daemon: termina con il processo principale)
        t = threading.Thread(target=self.tg.run_forever, daemon=True)
        t.start()

        try:
            while True:
                self._cycle()
                time.sleep(self.cfg.LOOP_SLEEP_SEC)
        except KeyboardInterrupt:
            self.log.info("Interruzione manuale — salvo stato e uscita")
        finally:
            self.pm.save()
            self.tg.send(f"🛑 Bot v2.0 terminato. Uptime: {self.tg._uptime()}")

    def _cycle(self):
        """Un ciclo completo su tutti i simboli."""
        equity = self.bc.get_usdc_balance()
        halted = self.rm.check_daily_drawdown(equity)
        win_rate = self.rm.estimate_win_rate(self.pm.trade_log)

        for symbol in self.cfg.SYMBOLS:
            try:
                self._process_symbol(symbol, equity, win_rate, halted)
            except Exception as e:
                self.log.error(f"Errore su {symbol}: {e}", exc_info=True)

    def _process_symbol(self, symbol: str, equity: float,
                         win_rate: float, halted: bool):
        # 1. Candele + indicatori
        df = self.bc.get_klines(symbol, self.cfg.INTERVAL,
                                 self.cfg.CANDLE_LIMIT)
        if df.empty or len(df) < self.cfg.EMA_TREND + 10:
            return
        df = self.ind.populate(df, self.cfg)

        position = self.pm.get(symbol)

        # ── GESTIONE POSIZIONE APERTA ─────────────────────────────────────
        if position:
            current_price = float(df.iloc[-1]["close"])
            # Aggiorna trailing stop
            position = self.rm.update_trailing_stop(position, current_price)
            self.pm.set(symbol, position)

            # Controlla uscita
            sell, reason = self.strat.check_sell(df, position)
            if sell:
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

        # ── VALUTAZIONE INGRESSO ──────────────────────────────────────────
        if halted or not self.tg.trading_enabled:
            return
        if self.pm.open_count() >= self.cfg.MAX_OPEN_POSITIONS:
            return

        # Order Book Imbalance (solo se non in posizione, risparmia API calls)
        ob_imb = self.bc.ob_imbalance(symbol)

        signal = self.strat.score_buy(df, ob_imb)
        last   = df.iloc[-1]
        self.log.debug(
            f"{symbol} score={signal.score:.1f} "
            f"(soglia={self.cfg.SIGNAL_THRESHOLD}) OBI={ob_imb:.2f}")

        if not signal.buy:
            return

        # Dimensione posizione (Kelly)
        invest = self.rm.kelly_position_size(
            win_rate, equity, signal.atr, last["close"])
        if invest < self.cfg.MIN_USDC_TRADE:
            self.log.info(
                f"{symbol}: Kelly amount {invest:.2f} < minimo. Skip.")
            return
        # Non superare il disponibile
        invest = min(invest, equity * 0.95)

        # Esegui BUY
        ref_price = last["close"]
        order = self.bc.buy_market_quote(symbol, invest, ref_price)
        if not order:
            return

        # Recupera dati ordine eseguito
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


# ══════════════════════════════════════════════════════════════════════════════
#  10. BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Backtest vettoriale su dati storici Binance.
    Simula commissioni (0.1%), slippage (0.05%) e Kelly position sizing.
    """

    COMMISSION = 0.001     # 0.1% per lato
    SLIPPAGE   = 0.0005    # 0.05% per lato

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.log  = build_logger(cfg.LOG_FILE)
        self.strat = Strategy(cfg)
        self.ind   = IndicatorEngine()
        self.rm    = RiskManager(cfg, self.log)

    def _fetch_klines(self, symbol: str, interval: str,
                      days: int) -> pd.DataFrame:
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
        wallet = initial_capital
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

            position = None
            trades   = []
            trade_log_local = []

            for i in range(self.cfg.EMA_TREND + 5, len(df)):
                window = df.iloc[:i + 1]
                row    = window.iloc[-1]
                price  = row["close"]
                date   = row["open_time"].strftime("%Y-%m-%d")

                # ── IN POSIZIONE: aggiorna trailing stop, controlla uscita
                if position:
                    # Trailing stop
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
                            "symbol": symbol,
                            "entry":  position["entry_price"],
                            "exit":   exit_p,
                            "qty":    position["qty"],
                            "pnl":    pnl,
                            "pnl_pct": pnl / position["invested"] * 100,
                            "reason": reason,
                            "date":   date,
                        })
                        trade_log_local.append({"pnl": pnl})
                        emoji = "🟢" if pnl > 0 else "🔴"
                        print(f"  {emoji} {date} SELL @ {exit_p:.4f} | "
                              f"PnL={pnl:+.2f} | {reason}")
                        position = None
                    continue

                # ── FUORI POSIZIONE: valuta ingresso
                # OB imbalance non disponibile in backtest → 0.5 (neutro)
                signal = self.strat.score_buy(window, ob_imbalance=0.5)
                if not signal.buy:
                    continue

                wr      = self.rm.estimate_win_rate(trade_log_local)
                invest  = self.rm.kelly_position_size(
                    wr, wallet, signal.atr, price)
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
                print(f"\n  Totale trade: {len(trades)} | Wins: {wins} "
                      f"({wins/len(trades)*100:.0f}%) | PnL: {tot_pnl:+.2f} USDC")

        # Riepilogo globale
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
    parser = argparse.ArgumentParser(description="Crypto Trading Bot v2.0")
    parser.add_argument("--live",      action="store_true", help="Trading live su Binance")
    parser.add_argument("--backtest",  action="store_true", help="Esegui backtest storico")
    parser.add_argument("--days",      type=int,   default=500,  help="Giorni backtest")
    parser.add_argument("--capital",   type=float, default=100.0, help="Capitale backtest (USDC)")
    parser.add_argument("--symbols",   type=str, default=None,
                        help="Simboli separati da virgola (es. BTCUSDC,ETHUSDC)")
    parser.add_argument("--interval",  type=str, default=None,
                        help="Intervallo candele (es. 1h, 4h, 1d)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Soglia score segnale (default 60)")
    args = parser.parse_args()

    # ── Config base — INSERISCI QUI LE TUE CREDENZIALI ──────────────────
    cfg = Config(

    )

    # Override da riga di comando
    if args.symbols:
        cfg.SYMBOLS = [s.strip().upper() for s in args.symbols.split(",")]
    if args.interval:
        cfg.INTERVAL = args.interval
    if args.threshold is not None:
        cfg.SIGNAL_THRESHOLD = args.threshold

    if args.live:
        bot = TradingBot(cfg)
        bot.start()

    elif args.backtest:
        bt = Backtester(cfg)
        bt.run(days=args.days, initial_capital=args.capital)

    else:
        parser.print_help()



