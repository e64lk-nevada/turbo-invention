#!/usr/bin/env python3
"""
Polymarket 5m Signals
Open + Now = Chainlink via Polymarket RTDS (wss://ws-live-data.polymarket.com)
Fallback = Binance
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List
from collections import deque

import httpx
import websockets
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT").split(",")]
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.78"))
WINDOW_SECONDS = 300
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
CONFIRM_AT = 180          # entry only after 3 min
PRICE_HISTORY_LEN = 60
MIN_LEAD_BPS = 10.0        # stronger lead filter (was 8)
MIN_MOM_SHORT = 4.0        # short momentum confirmation
MIN_MOM_MED = 2.5          # medium momentum confirmation

# Chainlink symbol map for RTDS
CHAINLINK_SYM = {
    "BTCUSDT": "btc/usd",
    "ETHUSDT": "eth/usd",
    "SOLUSDT": "sol/usd",
    "XRPUSDT": "xrp/usd",
}
SYM_FROM_CL = {v: k for k, v in CHAINLINK_SYM.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class WindowState:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.price_to_beat: Optional[float] = None
        self.window_start: Optional[datetime] = None
        self.prices: deque = deque(maxlen=PRICE_HISTORY_LEN)
        self.last_price: Optional[float] = None
        self.signal_sent: bool = False
        self.last_signal: Optional[dict] = None
        self.last_update: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.source: str = ""
        self.open_confirmed: bool = False  # True when open taken from Polymarket

    def reset(self, price: float, start: datetime, source: str = ""):
        self.price_to_beat = price
        self.window_start = start
        self.prices.clear()
        self.prices.append(price)
        self.last_price = price
        self.signal_sent = False
        self.last_signal = None
        self.last_update = datetime.now(timezone.utc)
        self.last_error = None
        self.source = source
        self.open_confirmed = False


state: Dict[str, WindowState] = {s: WindowState(s) for s in SYMBOLS}
signal_history: List[dict] = []
global_error: Optional[str] = None
live_chainlink: Dict[str, float] = {}  # symbol -> last chainlink price


def momentum_bps(prices: deque, lookback: int = 12) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    return (prices[-1] - prices[-lookback]) / prices[-lookback] * 10000


def generate_signal(price_to_beat: float, current_price: float, seconds_passed: int, prices: deque) -> dict:
    """
    Multi-factor signal inspired by jury-style bots:
    - Judge 1: Lead vs priceToBeat (Chainlink open)
    - Judge 2: Short momentum (last ~12 ticks)
    - Judge 3: Medium momentum (last ~25 ticks)
    Only fire when majority agree + strong enough edge.
    """
    if seconds_passed < CONFIRM_AT:
        return {"side": None, "confidence": 0.0, "reason": "too_early", "score": 0}

    lead_bps = ((current_price - price_to_beat) / price_to_beat) * 10000
    time_left = WINDOW_SECONDS - seconds_passed
    abs_lead = abs(lead_bps)

    if abs_lead < MIN_LEAD_BPS:
        return {"side": None, "confidence": 0.0, "reason": "too_small_lead", "score": 0, "lead_bps": round(lead_bps, 2)}

    mom_short = momentum_bps(prices, lookback=12)
    mom_med = momentum_bps(prices, lookback=25)

    side = "UP" if lead_bps > 0 else "DOWN"

    # Jury votes
    votes = 0
    # Judge 1: lead
    if abs_lead >= MIN_LEAD_BPS:
        votes += 1
    # Judge 2: short momentum same direction
    if (side == "UP" and mom_short >= MIN_MOM_SHORT) or (side == "DOWN" and mom_short <= -MIN_MOM_SHORT):
        votes += 1
    # Judge 3: medium momentum same direction
    if (side == "UP" and mom_med >= MIN_MOM_MED) or (side == "DOWN" and mom_med <= -MIN_MOM_MED):
        votes += 1

    if votes < 2:
        return {
            "side": None,
            "confidence": 0.0,
            "reason": f"jury_{votes}/3",
            "score": votes,
            "lead_bps": round(lead_bps, 2),
            "mom_short": round(mom_short, 2),
            "mom_med": round(mom_med, 2),
        }

    # Confidence curve: later + stronger lead = higher conf
    base = 0.72 + (votes - 2) * 0.06          # 2 votes → 0.72, 3 votes → 0.78
    lead_boost = min(abs_lead * 0.011, 0.14)
    time_boost = 0.0
    if time_left <= 45:
        time_boost = 0.08
    elif time_left <= 70:
        time_boost = 0.04

    confidence = min(base + lead_boost + time_boost, 0.96)

    return {
        "side": side,
        "confidence": round(confidence, 3),
        "lead_bps": round(lead_bps, 2),
        "time_left": time_left,
        "score": votes,
        "mom_short": round(mom_short, 2),
        "mom_med": round(mom_med, 2),
        "reason": f"jury_{votes}/3",
    }


def is_strong_signal(signal: dict) -> bool:
    if signal.get("side") is None:
        return False
    return signal.get("confidence", 0) >= MIN_CONFIDENCE and signal.get("score", 0) >= 2


def get_window_start_unix() -> int:
    now = int(datetime.now(timezone.utc).timestamp())
    return now - (now % WINDOW_SECONDS)


def get_current_window_start() -> datetime:
    return datetime.fromtimestamp(get_window_start_unix(), tz=timezone.utc)


async def fetch_binance(client: httpx.AsyncClient, symbol: str) -> Optional[float]:
    for url in [
        f"https://data-api.binance.vision/api/v3/ticker/price?symbol={symbol}",
        f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}",
    ]:
        try:
            r = await client.get(url, timeout=4.0)
            if r.status_code == 200:
                return float(r.json()["price"])
        except Exception:
            continue
    return None


async def fetch_polymarket_open(client: httpx.AsyncClient, symbol: str, start_unix: int) -> Optional[float]:
    """Price to Beat straight from Polymarket (same endpoint the website uses)."""
    base = symbol.replace("USDT", "")
    start_iso = datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = datetime.fromtimestamp(start_unix + WINDOW_SECONDS, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = await client.get(
            "https://polymarket.com/api/crypto/crypto-price",
            params={
                "symbol": base,
                "eventStartTime": start_iso,
                "variant": "fiveminute",
                "endDate": end_iso,
            },
            timeout=3.0,
        )
        if r.status_code == 200:
            val = r.json().get("openPrice")
            if val is not None:
                return float(val)
    except Exception as e:
        logger.debug(f"polymarket open fetch {symbol}: {e}")
    return None


async def fetch_polymarket_close(client: httpx.AsyncClient, symbol: str, start_unix: int) -> Optional[float]:
    """Final price of a finished window straight from Polymarket."""
    base = symbol.replace("USDT", "")
    start_iso = datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = datetime.fromtimestamp(start_unix + WINDOW_SECONDS, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = await client.get(
            "https://polymarket.com/api/crypto/crypto-price",
            params={"symbol": base, "eventStartTime": start_iso, "variant": "fiveminute", "endDate": end_iso},
            timeout=3.0,
        )
        if r.status_code == 200:
            val = r.json().get("closePrice")
            if val is not None:
                return float(val)
    except Exception as e:
        logger.debug(f"polymarket close fetch {symbol}: {e}")
    return None


async def send_telegram(client: httpx.AsyncClient, text: str):
    """Send a message to Telegram (only if token + chat id are set)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
            timeout=8.0,
        )
        if r.status_code != 200:
            logger.warning(f"Telegram error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ====================== RTDS Chainlink ======================
async def rtds_loop():
    """Subscribe to Polymarket RTDS Chainlink prices."""
    url = "wss://ws-live-data.polymarket.com"
    while True:
        try:
            async with websockets.connect(url, ping_interval=None, close_timeout=5) as ws:
                # one subscription for ALL chainlink symbols (no filters) - most reliable
                subs = [{
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": "",
                }]
                await ws.send(json.dumps({"action": "subscribe", "subscriptions": subs}))
                logger.info("RTDS subscribed to Chainlink prices")

                async def ping():
                    while True:
                        await asyncio.sleep(5)
                        try:
                            await ws.send("PING")
                        except Exception:
                            break

                ping_task = asyncio.create_task(ping())
                try:
                    async for raw in ws:
                        if raw == "PONG":
                            continue
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        topic = msg.get("topic") or ""
                        if "chainlink" not in topic and topic != "crypto_prices_chainlink":
                            continue
                        payload = msg.get("payload") or {}
                        sym = (payload.get("symbol") or "").lower()
                        val = payload.get("value")
                        if val is None and isinstance(payload.get("data"), list) and payload["data"]:
                            val = payload["data"][-1].get("value")  # initial snapshot
                        if val is None:
                            continue
                        usdt = SYM_FROM_CL.get(sym)
                        if not usdt:
                            continue
                        price = float(val)
                        if usdt not in live_chainlink:
                            logger.info(f"RTDS first chainlink price {usdt}={price}")
                        live_chainlink[usdt] = price
                finally:
                    ping_task.cancel()
        except Exception as e:
            logger.warning(f"RTDS reconnect in 3s: {e}")
            await asyncio.sleep(3)


async def price_loop():
    global global_error
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        while True:
            try:
                start_unix = get_window_start_unix()
                window_start = get_current_window_start()
                seconds_passed = int((datetime.now(timezone.utc) - window_start).total_seconds())
                any_ok = False

                for symbol in SYMBOLS:
                    ws = state[symbol]

                    # prefer Chainlink live price
                    price = live_chainlink.get(symbol)
                    source_now = "chainlink"
                    if price is None:
                        price = await fetch_binance(client, symbol)
                        source_now = "binance"

                    if price is None:
                        ws.last_error = "no price"
                        continue

                    any_ok = True
                    ws.last_error = None

                    # new window
                    
                    # close previous window: mark history outcomes ONLY when the window has ended
                    if (ws.window_start is not None and ws.window_start != window_start
                            and ws.last_signal and ws.price_to_beat and ws.last_price):
                        prev_start_unix = int(ws.window_start.timestamp())
                        final = await fetch_polymarket_close(client, symbol, prev_start_unix)
                        if final is None:
                            final = ws.last_price
                        open_p = ws.price_to_beat
                        actual = "UP" if final >= open_p else "DOWN"
                        predicted = ws.last_signal.get("side")
                        for h in signal_history:
                            if h.get("window_start") == (ws.window_start.isoformat() if hasattr(ws.window_start, "isoformat") else str(ws.window_start)) and h.get("symbol") == symbol.replace("USDT",""):
                                h["result"] = "win" if predicted == actual else "loss"
                                h["final"] = final
                                h["actual"] = actual
                                break

                    if ws.window_start is None or ws.window_start != window_start:
                        # capture open from Chainlink if available, else binance
                        pm_open = await fetch_polymarket_open(client, symbol, start_unix)
                        if pm_open is not None:
                            open_px, open_src = pm_open, "polymarket"
                        else:
                            open_px = live_chainlink.get(symbol) or price
                            open_src = "chainlink" if symbol in live_chainlink else "binance-fallback"
                        ws.reset(open_px, window_start, source=open_src)
                        ws.open_confirmed = pm_open is not None
                        logger.info(f"[{symbol}] New window open={open_px:.4f} ({open_src})")
                        continue

                    # Polymarket open not available at window start yet -> keep retrying
                    if not ws.open_confirmed:
                        pm_open = await fetch_polymarket_open(client, symbol, start_unix)
                        if pm_open is not None:
                            ws.price_to_beat = pm_open
                            ws.open_confirmed = True
                            ws.source = "polymarket"
                            logger.info(f"[{symbol}] Open set from Polymarket: {pm_open:.4f}")

                    # update live
                    ws.prices.append(price)
                    ws.last_price = price
                    ws.last_update = datetime.now(timezone.utc)
                    if source_now == "chainlink" and ws.source != "chainlink" and not ws.open_confirmed:
                        # upgrade open if we only had binance at start and still early
                        if seconds_passed < 15 and ws.source.startswith("binance"):
                            ws.price_to_beat = price  # rare: only first seconds
                        ws.source = "chainlink" if ws.source.startswith("chainlink") or source_now == "chainlink" else ws.source

                    if seconds_passed < CONFIRM_AT or ws.signal_sent:
                        continue

                    signal = generate_signal(ws.price_to_beat, price, seconds_passed, ws.prices)
                    if is_strong_signal(signal):
                        ws.signal_sent = True
                        entry = {
                            **signal,
                            "symbol": symbol.replace("USDT", ""),
                            "momentum_bps": signal.get("mom_short", 0),
                            "price": price,
                            "open": ws.price_to_beat,
                            "time": datetime.now(timezone.utc).isoformat(),
                            "window_start": window_start.isoformat(),
                        }
                        ws.last_signal = entry
                        signal_history.insert(0, entry)
                        if len(signal_history) > 50:
                            signal_history.pop()
                        name = symbol.replace("USDT", "")
                        arrow = "🟢 ВГОРУ (UP)" if signal["side"] == "UP" else "🔴 ВНИЗ (DOWN)"
                        asyncio.create_task(send_telegram(client, (
                            f"⚡ {name} 5m сигнал\n"
                            f"{arrow}  conf {signal['confidence'] * 100:.0f}%  ({signal.get('score')}/3)\n"
                            f"Open: {ws.price_to_beat}\n"
                            f"Now: {price}\n"
                            f"Lead: {signal.get('lead_bps')} bps\n"
                            f"Залишилось: {signal.get('time_left')}s\n"
                            f"https://polymarket.com/event/{name.lower()}-updown-5m-{start_unix}"
                        )))
                        logger.info(
                            f"SIGNAL {symbol} {signal['side']} conf={signal['confidence']} "
                            f"score={signal.get('score')} lead={signal.get('lead_bps')} "
                            f"mom={signal.get('mom_short')}"
                        )

                global_error = None if any_ok else "Немає цін"
                await asyncio.sleep(1.5)
            except Exception as e:
                global_error = str(e)
                logger.error(f"Loop: {e}")
                await asyncio.sleep(5)


async def api_status(request):
    window_start = get_current_window_start()
    seconds_passed = int((datetime.now(timezone.utc) - window_start).total_seconds())
    time_left = max(0, WINDOW_SECONDS - seconds_passed)

    markets = {}
    for symbol, ws in state.items():
        name = symbol.replace("USDT", "")
        if ws.price_to_beat is None:
            markets[name] = {"status": "waiting", "error": ws.last_error}
            continue
        current = ws.last_price or ws.price_to_beat
        lead = ((current - ws.price_to_beat) / ws.price_to_beat) * 10000
        live = generate_signal(ws.price_to_beat, current, max(seconds_passed, CONFIRM_AT), ws.prices)
        markets[name] = {
            "open": round(ws.price_to_beat, 4),
            "current": round(current, 4),
            "lead_bps": round(lead, 2),
            "signal_sent": ws.signal_sent,
            "signal": ws.last_signal,
            "live_side": live.get("side"),
            "live_confidence": live.get("confidence", 0),
            "live_score": live.get("score", 0),
            "source": ws.source,
            "updated": ws.last_update.isoformat() if ws.last_update else None,
        }

    # simple win-rate from resolved history
    resolved = [h for h in signal_history if h.get("result") in ("win", "loss")]
    wins = sum(1 for h in resolved if h["result"] == "win")
    losses = sum(1 for h in resolved if h["result"] == "loss")
    total_r = wins + losses
    win_rate = round(wins / total_r * 100, 1) if total_r else None

    return JSONResponse({
        "server_time": datetime.now(timezone.utc).isoformat(),
        "window_start": window_start.isoformat(),
        "seconds_passed": seconds_passed,
        "time_left": time_left,
        "confirm_at": CONFIRM_AT,
        "error": global_error,
        "chainlink_live": {k.replace("USDT",""): v for k,v in live_chainlink.items()},
        "markets": markets,
        "history": signal_history[:50],
        "stats": {
            "signals": len(signal_history),
            "resolved": total_r,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
        },
    })


async def health(request):
    return JSONResponse({"ok": True, "chainlink": list(live_chainlink.keys())})


# HTML — same dashboard (compact)
HTML_PAGE = """<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<title>5m Signals</title>
<style>
:root{--bg:#0a0e17;--card:#121826;--border:#1c2436;--text:#e8eef8;--muted:#7b8ba5;--up:#22c55e;--down:#ef4444;--accent:#5b6cff}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:14px;padding-bottom:40px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
h1{font-size:1.15rem;font-weight:700}
.badge{font-size:.72rem;background:#1a2235;color:var(--muted);padding:4px 10px;border-radius:999px;border:1px solid var(--border)}
.badge.ok{color:var(--up);border-color:rgba(34,197,94,.3)}
.top{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.ac{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:12px 14px}
.an{font-size:.72rem;color:var(--muted)}.ap{font-size:1.05rem;font-weight:700;font-variant-numeric:tabular-nums}
.ach{font-size:.78rem;font-weight:600}.pos{color:var(--up)}.neg{color:var(--down)}
.panel{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:14px;margin-bottom:12px}
.pt{font-size:.8rem;color:var(--muted);margin-bottom:10px;display:flex;justify-content:space-between}
.mrow{display:grid;grid-template-columns:50px 1fr 1fr 90px;gap:6px;padding:9px 0;border-bottom:1px solid var(--border);font-size:.88rem;align-items:center}
.mrow.h{color:var(--muted);font-size:.7rem;text-transform:uppercase;border:none}
.bs{padding:3px 8px;border-radius:6px;font-size:.72rem;font-weight:700}
.bu{background:rgba(34,197,94,.15);color:var(--up)}.bd{background:rgba(239,68,68,.15);color:var(--down)}.bw{background:#1a2235;color:var(--muted)}
.timer{font-size:2rem;font-weight:700;font-variant-numeric:tabular-nums;margin:6px 0 12px}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.bup,.bdn{padding:13px;border-radius:12px;border:none;font-weight:700;font-size:.9rem;text-align:center;opacity:.45;transition:.2s}
.bup{background:rgba(34,197,94,.18);color:var(--up)}.bdn{background:rgba(239,68,68,.18);color:var(--down)}
.bup.on,.bdn.on{opacity:1;box-shadow:0 0 0 2px currentColor}
.conf{height:6px;background:#1a2235;border-radius:4px;overflow:hidden;margin-top:6px}
.conf i{display:block;height:100%;background:var(--accent);width:0;transition:width .4s}
.ct{font-size:.85rem;color:var(--muted);margin-top:4px}
.lr{display:flex;justify-content:space-between;padding:5px 0;font-size:.85rem;border-bottom:1px solid var(--border)}
.hrow{display:grid;grid-template-columns:60px 40px 30px 40px 1fr;gap:4px;padding:7px 0;border-bottom:1px solid var(--border);font-size:.78rem;align-items:center}
.stats{display:flex;gap:12px;font-size:.78rem;color:var(--muted);margin-bottom:8px;flex-wrap:wrap}
.stats b{color:var(--text)}
.acc-sum{font-size:.95rem;line-height:1.5;margin-bottom:12px}
.acc-sum b{font-family:ui-monospace,Menlo,monospace}
.sqs{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:14px}
.sq{width:18px;height:18px;border-radius:4px;background:#1c2436}
.sq.w{background:#2fa87b}.sq.l{background:#c0564f}
.cnt{display:flex;gap:26px;margin-bottom:10px}
.cnt b{display:block;font-size:1.6rem;font-weight:700;font-family:ui-monospace,Menlo,monospace}
.cnt span{font-size:.65rem;letter-spacing:.14em;color:var(--muted)}
.tgl{font-size:.68rem;letter-spacing:.14em;color:var(--muted);background:#121826;border:1px solid var(--border);border-radius:999px;padding:7px 14px;cursor:pointer;text-transform:uppercase}
.hl{display:none;max-height:430px;overflow-y:auto;border-top:1px solid var(--border);margin-top:12px}
.hl.open{display:block}
.hr2{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid var(--border);font-family:ui-monospace,Menlo,monospace;font-size:.8rem}
.hr2 .tx{flex:1}.hr2 .tm{color:var(--muted)}
.ic{width:24px;height:24px;border-radius:6px;border:1px solid;display:flex;align-items:center;justify-content:center;font-size:.8rem;flex:none}
.ic.w{color:#2fa87b;border-color:#2fa87b;background:rgba(47,168,123,.12)}
.ic.l{color:#c0564f;border-color:#c0564f;background:rgba(192,86,79,.12)}
.ic.p{color:var(--muted);border-color:var(--border)}
</style>
</head>
<body>
<header>
  <h1>⚡ 5m Signals</h1>
  <div id="badge" class="badge">…</div>
</header>
<div class="top" id="top"></div>
<div class="panel">
  <div class="pt"><span>Ринки</span><span id="tmr"></span></div>
  <div class="mrow h"><div>Актив</div><div>Open</div><div>Now</div><div>Сигнал</div></div>
  <div id="mkt"></div>
</div>
<div class="panel">
  <div class="pt"><span>Live BTC</span><span id="ct">—</span></div>
  <div class="btns">
    <div id="bu" class="bup">ВГОРУ</div>
    <div id="bd" class="bdn">ВНИЗ</div>
  </div>
  <div class="conf"><i id="cf"></i></div>
  <div id="ll" style="margin-top:10px"></div>
</div>
<div class="panel">
  <div class="pt" style="align-items:center"><span style="letter-spacing:.16em;text-transform:uppercase">Остання точність</span><button class="tgl" id="tgl">Історія ▾</button></div>
  <div class="acc-sum" id="accsum">—</div>
  <div class="sqs" id="sqs"></div>
  <div class="cnt" id="cnt"></div>
  <div class="hl" id="hl"></div>
</div>
<script>
const S=["BTC","ETH","SOL","XRP"];
function f(n,d=2){return n==null?"—":Number(n).toFixed(d)}
function render(d){
  const badge=document.getElementById("badge");
  if(d.error){badge.textContent=d.error;badge.className="badge"}
  else{badge.textContent="live · "+(d.seconds_passed||0)+"s";badge.className="badge ok"}
  document.getElementById("tmr").textContent=(d.time_left||0)+"s left · conf@"+d.confirm_at+"s";
  let top="";
  for(const n of S){
    const m=d.markets[n];
    if(!m||m.status==="waiting"){top+=`<div class="ac"><div class="an">${n}</div><div class="ap">—</div></div>`;continue}
    const c=m.lead_bps>=0?"pos":"neg";
    top+=`<div class="ac"><div class="an">${n} · ${m.source||""}</div><div class="ap">$${f(m.current,n==="XRP"?4:2)}</div><div class="ach ${c}">${m.lead_bps>=0?"+":""}${m.lead_bps} bps</div></div>`;
  }
  document.getElementById("top").innerHTML=top;
  let body="";
  for(const n of S){
    const m=d.markets[n];
    if(!m||m.status==="waiting"){body+=`<div class="mrow"><div>${n}</div><div>—</div><div>—</div><div class="bs bw">…</div></div>`;continue}
    let sh=`<span class="bs bw">очікування</span>`;
    if(m.signal_sent&&m.signal){
      const s=m.signal.side;
      const sc=m.signal.score?` ${m.signal.score}/3`:"";
      sh=`<span class="bs ${s==="UP"?"bu":"bd"}">${s==="UP"?"ВГОРУ":"ВНИЗ"} ${(m.signal.confidence*100).toFixed(0)}%${sc}</span>`;
    }else if(d.seconds_passed>=d.confirm_at&&m.live_side){
      sh=`<span class="bs bw">${m.live_side} ~${(m.live_confidence*100).toFixed(0)}%</span>`;
    }
    body+=`<div class="mrow"><div style="font-weight:600">${n}</div><div>${f(m.open,n==="XRP"?4:2)}</div><div>${f(m.current,n==="XRP"?4:2)}</div><div>${sh}</div></div>`;
  }
  document.getElementById("mkt").innerHTML=body;
  const foc=d.markets.BTC||{};
  let conf=0,side=null;
  if(foc.signal_sent&&foc.signal){side=foc.signal.side;conf=foc.signal.confidence}
  else if(foc.live_side){side=foc.live_side;conf=foc.live_confidence||0}
  document.getElementById("bu").className="bup"+(side==="UP"?" on":"");
  document.getElementById("bd").className="bdn"+(side==="DOWN"?" on":"");
  document.getElementById("ct").textContent=conf?(conf*100).toFixed(1)+"%":"—";
  document.getElementById("cf").style.width=(conf*100)+"%";
  let live="";
  for(const n of S){
    const m=d.markets[n];if(!m||m.status==="waiting")continue;
    const c=m.signal_sent&&m.signal?m.signal.confidence:(m.live_confidence||0);
    const s=m.signal_sent&&m.signal?m.signal.side:m.live_side;
    live+=`<div class="lr"><span>${n}</span><span style="color:${s==="UP"?"var(--up)":s==="DOWN"?"var(--down)":"var(--muted)"}">${s||"—"} ${c?(c*100).toFixed(0)+"%":""}</span></div>`;
  }
  document.getElementById("ll").innerHTML=live;
  const hAll=(d.history||[]).slice();
  const sig=hAll.map(e=>e.time+(e.result||'p')).join('|');
  if(sig!==window._hsig){
    window._hsig=sig;
    const chron=hAll.slice().reverse();
    let sq='';
    for(const e of chron){sq+="<div class='sq "+(e.result==='win'?'w':e.result==='loss'?'l':'')+"'></div>"}
    document.getElementById('sqs').innerHTML=sq;
    const res=hAll.filter(e=>e.result==='win'||e.result==='loss');
    const w=res.filter(e=>e.result==='win').length,l=res.length-w;
    document.getElementById('accsum').innerHTML=res.length?"Бот вгадав <b>"+w+" з "+res.length+"</b> завершених вікон ("+(w/res.length*100).toFixed(1)+"%).":"Ще немає завершених вікон.";
    let streak=0;
    if(res.length){const first=res[0].result;for(const e of res){if(e.result===first)streak++;else break}streak=first==='win'?streak:-streak}
    document.getElementById('cnt').innerHTML="<div><b style='color:#2fa87b'>"+w+"</b><span>ПЕРЕМОГИ</span></div><div><b style='color:#c0564f'>"+l+"</b><span>ПРОГРАШІ</span></div><div><b>"+(streak>0?'+':'')+streak+"</b><span>СЕРІЯ</span></div>";
    let x='';
    for(const e of hAll){
      const t=e.time?new Date(e.time).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}):'—';
      const cls=e.result==='win'?'w':e.result==='loss'?'l':'p';
      const ic=e.result==='win'?'✓':e.result==='loss'?'✕':'·';
      const sc=e.side==='UP'?'var(--up)':'var(--down)';
      const ac=e.actual?"<span style='color:"+(e.actual==='UP'?'var(--up)':'var(--down)')+"'>"+e.actual+"</span>":"<span style='color:var(--muted)'>очікує</span>";
      x+="<div class='hr2'><div class='ic "+cls+"'>"+ic+"</div><div class='tx'>"+(e.symbol||'')+" сигнал <span style='color:"+sc+"'>"+e.side+"</span> · закрилось "+ac+"</div><div class='tm'>"+t+"</div></div>";
    }
    document.getElementById('hl').innerHTML=x||"<div class='hr2' style='color:var(--muted)'>Поки порожньо</div>";
  }
}
async function refresh(){try{render(await(await fetch("/api/status")).json())}catch(e){document.getElementById("badge").textContent="offline"}}
document.getElementById('tgl').onclick=function(){const h=document.getElementById('hl');const o=h.classList.toggle('open');this.textContent=o?'Сховати ▴':'Історія ▾'};
refresh();setInterval(refresh,2000);
</script>
</body>
</html>"""


async def dashboard(request):
    return HTMLResponse(HTML_PAGE)


async def on_startup():
    asyncio.create_task(rtds_loop())
    asyncio.create_task(price_loop())
    logger.info("RTDS + price loops started")


app = Starlette(
    routes=[
        Route("/", dashboard),
        Route("/api/status", api_status),
        Route("/health", health),
    ],
    on_startup=[on_startup],
)
