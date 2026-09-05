from __future__ import annotations

import asyncio
import math
import os
from datetime import datetime, timezone
from typing import Any

import ccxt.async_support as ccxt
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# -------------------- configuration --------------------
BOT_MODE = os.getenv("BOT_MODE", "paper").strip().lower()  # paper | testnet only
EXECUTE_TESTNET_ORDERS = os.getenv("EXECUTE_TESTNET_ORDERS", "false").strip().lower() in {"1", "true", "yes", "on"}
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

DEFAULT_SYMBOLS = "BTC/USDT,ETH/USDT,BNB/USDT" if BOT_MODE == "testnet" else "BTC/USDC,ETH/USDC,BNB/USDC"
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]

ENTRY_SCORE = float(os.getenv("ENTRY_SCORE", "78"))
LOOP_SECONDS = max(30, int(os.getenv("LOOP_SECONDS", "60")))
COOLDOWN_MINUTES = max(15, int(os.getenv("COOLDOWN_MINUTES", "45")))
START_BAL = float(os.getenv("STARTING_PAPER_BALANCE", "1000"))
MAX_FRACTION = min(max(float(os.getenv("MAX_CAPITAL_FRACTION_PER_TRADE", "0.10")), 0.01), 0.25)
MAX_QUOTE_PER_TRADE = max(5.0, float(os.getenv("MAX_QUOTE_PER_TRADE", "50")))
MIN_QUOTE_PER_TRADE = max(5.0, float(os.getenv("MIN_QUOTE_PER_TRADE", "5")))
MAX_DAILY_LOSS_PCT = min(max(float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0")), 0.5), 3.0)
STOP_ATR = min(max(float(os.getenv("STOP_ATR_MULT", "1.6")), 0.8), 3.0)
TP_ATR = min(max(float(os.getenv("TAKE_PROFIT_ATR_MULT", "2.8")), 1.2), 6.0)
TRAIL_ATR = min(max(float(os.getenv("TRAILING_ATR_MULT", "1.1")), 0.5), 3.0)
TRAIL_ACTIVATE = min(max(float(os.getenv("TRAILING_ACTIVATE_ATR", "1.2")), 0.5), 3.0)
MAX_HOLD_HOURS = max(1, int(os.getenv("MAX_HOLD_HOURS", "24")))

if BOT_MODE not in {"paper", "testnet"}:
    raise RuntimeError("Safety lock: BOT_MODE supports only 'paper' or 'testnet'. Live/mainnet is intentionally unavailable.")

app = FastAPI(title="Khafi Spot Bot v3")

exchange_args: dict[str, Any] = {
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
}
if BOT_MODE == "testnet" and API_KEY and API_SECRET:
    exchange_args.update({"apiKey": API_KEY, "secret": API_SECRET})

ex = ccxt.binance(exchange_args)
if BOT_MODE == "testnet":
    # Must be called immediately after construction. This is the hard sandbox boundary.
    ex.set_sandbox_mode(True)

state: dict[str, Any] = {
    "version": "v3",
    "mode": BOT_MODE,
    "execution_enabled": EXECUTE_TESTNET_ORDERS if BOT_MODE == "testnet" else False,
    "status": "starting",
    "paper_balance": START_BAL,
    "realized_pnl": 0.0,
    "day_start_equity": START_BAL,
    "day_key": None,
    "position": None,
    "last_signal": None,
    "last_trade_at": None,
    "trades": [],
    "errors": [],
    "stats": {"wins": 0, "losses": 0, "closed": 0, "gross_profit": 0.0, "gross_loss": 0.0},
    "preflight": {"ok": False, "symbols": {}, "auth_configured": bool(API_KEY and API_SECRET)},
}

task: asyncio.Task | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ema_series(values: list[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def rsi_at(values: list[float], period: int, idx: int) -> float:
    if idx < period:
        return 50.0
    gains, losses = [], []
    for i in range(idx - period + 1, idx + 1):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains) / period
    al = sum(losses) / period
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))


def atr_at(rows: list[list[Any]], period: int, idx: int) -> float:
    if idx < 1:
        return 0.0
    trs = []
    for i in range(max(1, idx - period + 1), idx + 1):
        high = float(rows[i][2])
        low = float(rows[i][3])
        prev = float(rows[i - 1][4])
        trs.append(max(high - low, abs(high - prev), abs(low - prev)))
    return sum(trs) / max(len(trs), 1)


def volume_ratio_at(rows: list[list[Any]], period: int, idx: int) -> float:
    start = max(0, idx - period + 1)
    vals = [float(rows[i][5]) for i in range(start, idx + 1)]
    avg = sum(vals) / max(len(vals), 1)
    return float(rows[idx][5]) / avg if avg > 0 else 0.0


def ema_slope_pct(series: list[float], lookback: int = 6) -> float:
    if len(series) <= lookback or series[-lookback] == 0:
        return 0.0
    return (series[-1] / series[-lookback] - 1.0) * 100.0


async def ohlcv(symbol: str, tf: str):
    return await ex.fetch_ohlcv(symbol, timeframe=tf, limit=250)


def analyze(symbol: str, d15: list[list[Any]], d1h: list[list[Any]]) -> dict[str, Any]:
    if len(d15) < 220 or len(d1h) < 220:
        return {"time": now_iso(), "symbol": symbol, "score": 0, "eligible": False, "reasons": ["not_enough_data"]}

    i15 = len(d15) - 2  # only fully closed candle
    i1h = len(d1h) - 2
    c15 = [float(r[4]) for r in d15[: i15 + 1]]
    c1h = [float(r[4]) for r in d1h[: i1h + 1]]

    e20s = ema_series(c15, 20)
    e50s = ema_series(c15, 50)
    h50s = ema_series(c1h, 50)
    h200s = ema_series(c1h, 200)

    price = c15[-1]
    e20, e50 = e20s[-1], e50s[-1]
    h50, h200 = h50s[-1], h200s[-1]
    hprice = c1h[-1]
    rv = rsi_at(c15, 14, len(c15) - 1)
    av = atr_at(d15, 14, i15)
    vr = volume_ratio_at(d15, 20, i15)
    vol_pct = (av / price * 100.0) if price > 0 else 0.0
    slope15 = ema_slope_pct(e20s, 6)
    slope1h = ema_slope_pct(h50s, 6)

    score = 0.0
    reasons: list[str] = []

    # Hard regime filter: do not buy against the 1h primary trend.
    regime_up = h50 > h200 and hprice > h50 and slope1h > 0
    if regime_up:
        score += 32
        reasons.append("1h_primary_uptrend")
    else:
        reasons.append("1h_regime_rejected")

    structure_up = e20 > e50 and price > e20 and slope15 > 0
    if structure_up:
        score += 23
        reasons.append("15m_structure_up")
    elif e20 > e50:
        score += 10
        reasons.append("15m_structure_partial")

    if 48 <= rv <= 63:
        score += 15
        reasons.append(f"rsi_sweetspot:{rv:.1f}")
    elif 43 <= rv < 48:
        score += 8
        reasons.append(f"rsi_early:{rv:.1f}")
    elif rv >= 70:
        score -= 18
        reasons.append(f"rsi_overbought:{rv:.1f}")
    else:
        reasons.append(f"rsi_neutral:{rv:.1f}")

    if vr >= 1.20:
        score += 15
        reasons.append(f"volume_strong:{vr:.2f}")
    elif vr >= 0.90:
        score += 8
        reasons.append(f"volume_ok:{vr:.2f}")
    else:
        reasons.append(f"volume_weak:{vr:.2f}")

    extension_atr = abs(price - e20) / av if av > 0 else 99.0
    if extension_atr <= 0.75:
        score += 10
        reasons.append(f"entry_not_extended:{extension_atr:.2f}ATR")
    elif extension_atr > 1.6:
        score -= 15
        reasons.append(f"entry_overextended:{extension_atr:.2f}ATR")

    volatility_ok = 0.12 <= vol_pct <= 2.5
    if volatility_ok:
        score += 5
        reasons.append(f"volatility_ok:{vol_pct:.2f}%")
    else:
        reasons.append(f"volatility_rejected:{vol_pct:.2f}%")

    score = clamp(score, 0, 100)
    eligible = bool(regime_up and structure_up and volatility_ok and av > 0 and score >= ENTRY_SCORE)

    return {
        "time": now_iso(),
        "symbol": symbol,
        "score": round(score, 1),
        "eligible": eligible,
        "price": price,
        "atr": av,
        "rsi": round(rv, 2),
        "volume_ratio": round(vr, 2),
        "volatility_pct": round(vol_pct, 3),
        "ema20_slope_pct": round(slope15, 3),
        "ema50_1h_slope_pct": round(slope1h, 3),
        "reasons": reasons,
        "stop": price - STOP_ATR * av,
        "tp": price + TP_ATR * av,
    }


def cooldown_blocked() -> bool:
    if not state["last_trade_at"]:
        return False
    try:
        t = datetime.fromisoformat(state["last_trade_at"])
        return (datetime.now(timezone.utc) - t).total_seconds() / 60.0 < COOLDOWN_MINUTES
    except Exception:
        return False


def daily_loss_pct() -> float:
    base = max(float(state["day_start_equity"]), 1e-9)
    return max(0.0, -float(state["realized_pnl"])) / base * 100.0


async def paper_enter(sig: dict[str, Any]):
    alloc = min(float(state["paper_balance"]) * MAX_FRACTION, MAX_QUOTE_PER_TRADE)
    if alloc < MIN_QUOTE_PER_TRADE:
        state["status"] = f"paper_capital_too_small:{alloc:.2f}"
        return
    base = alloc / sig["price"]
    state["paper_balance"] -= alloc
    state["position"] = {
        "symbol": sig["symbol"], "entry": sig["price"], "base": base, "spent": alloc,
        "atr": sig["atr"], "stop": sig["stop"], "tp": sig["tp"], "highest": sig["price"],
        "opened": now_iso(), "trail": False, "venue": "paper",
    }
    state["last_trade_at"] = now_iso()
    state["trades"].append({
        "time": now_iso(), "type": "PAPER_BUY", "symbol": sig["symbol"], "price": sig["price"],
        "quote": alloc, "score": sig["score"], "reasons": sig["reasons"],
    })


async def testnet_enter(sig: dict[str, Any]):
    if not EXECUTE_TESTNET_ORDERS:
        state["status"] = "testnet_signal_only"
        return
    if not (API_KEY and API_SECRET):
        state["status"] = "testnet_missing_api_credentials"
        return

    market = ex.market(sig["symbol"])
    quote = market["quote"]
    bal = await ex.fetch_balance()
    free_quote = float((bal.get("free") or {}).get(quote, 0) or 0)
    alloc = min(free_quote * MAX_FRACTION, MAX_QUOTE_PER_TRADE)
    if alloc < MIN_QUOTE_PER_TRADE:
        state["status"] = f"testnet_quote_too_small:{alloc:.2f}"
        return

    ticker = await ex.fetch_ticker(sig["symbol"])
    px = float(ticker.get("last") or sig["price"])
    amount = float(ex.amount_to_precision(sig["symbol"], alloc / px))
    if amount <= 0:
        state["status"] = "testnet_bad_order_amount"
        return

    order = await ex.create_market_buy_order(sig["symbol"], amount)
    filled = float(order.get("filled") or amount)
    avg = float(order.get("average") or px)
    if filled <= 0:
        state["status"] = "testnet_buy_not_filled"
        return

    state["position"] = {
        "symbol": sig["symbol"], "entry": avg, "base": filled, "spent": filled * avg,
        "atr": sig["atr"], "stop": avg - STOP_ATR * sig["atr"], "tp": avg + TP_ATR * sig["atr"],
        "highest": avg, "opened": now_iso(), "trail": False, "venue": "binance_spot_testnet",
        "order_id": order.get("id"),
    }
    state["last_trade_at"] = now_iso()
    state["trades"].append({"time": now_iso(), "type": "TESTNET_BUY", "symbol": sig["symbol"], "price": avg, "base": filled, "score": sig["score"]})


async def enter(sig: dict[str, Any]):
    if BOT_MODE == "paper":
        await paper_enter(sig)
    else:
        await testnet_enter(sig)


async def close_position(price: float, reason: str):
    p = state["position"]
    if not p:
        return

    if p.get("venue") == "binance_spot_testnet":
        if not EXECUTE_TESTNET_ORDERS:
            state["status"] = "testnet_exit_blocked_execution_disabled"
            return
        amount = float(ex.amount_to_precision(p["symbol"], p["base"]))
        await ex.create_market_sell_order(p["symbol"], amount)

    proceeds = float(p["base"]) * price
    pnl = proceeds - float(p["spent"])
    if p.get("venue") == "paper":
        state["paper_balance"] += proceeds

    state["realized_pnl"] += pnl
    st = state["stats"]
    st["closed"] += 1
    if pnl >= 0:
        st["wins"] += 1
        st["gross_profit"] += pnl
    else:
        st["losses"] += 1
        st["gross_loss"] += abs(pnl)

    state["trades"].append({"time": now_iso(), "type": "EXIT", "symbol": p["symbol"], "price": price, "pnl": pnl, "reason": reason, "venue": p.get("venue")})
    state["last_trade_at"] = now_iso()
    state["position"] = None


async def manage_position():
    p = state["position"]
    if not p:
        return
    t = await ex.fetch_ticker(p["symbol"])
    price = float(t["last"])
    p["highest"] = max(float(p["highest"]), price)

    if not p["trail"] and price >= p["entry"] + TRAIL_ACTIVATE * p["atr"]:
        p["trail"] = True

    dynamic_stop = float(p["stop"])
    if p["trail"]:
        dynamic_stop = max(dynamic_stop, p["highest"] - TRAIL_ATR * p["atr"])

    held = (datetime.now(timezone.utc) - datetime.fromisoformat(p["opened"])).total_seconds() / 3600.0
    if price <= dynamic_stop:
        await close_position(price, "stop_or_trailing")
    elif price >= p["tp"]:
        await close_position(price, "take_profit")
    elif held >= MAX_HOLD_HOURS:
        await close_position(price, "max_hold")


async def scan():
    if state["position"]:
        return
    if cooldown_blocked():
        state["status"] = "cooldown"
        return
    if daily_loss_pct() >= MAX_DAILY_LOSS_PCT:
        state["status"] = "daily_loss_lock"
        return

    best = None
    for sym in SYMBOLS:
        try:
            d15, d1h = await asyncio.gather(ohlcv(sym, "15m"), ohlcv(sym, "1h"))
            sig = analyze(sym, d15, d1h)
            if best is None or sig.get("score", 0) > best.get("score", 0):
                best = sig
        except Exception as e:
            state["errors"].append({"time": now_iso(), "symbol": sym, "error": str(e)[:220]})
            state["errors"] = state["errors"][-30:]

    if best:
        state["last_signal"] = best
        state["status"] = f"best:{best['symbol']}:{best['score']}"
        if best.get("eligible"):
            await enter(best)


async def run_preflight():
    pf = {"ok": True, "symbols": {}, "auth_configured": bool(API_KEY and API_SECRET), "sandbox": BOT_MODE == "testnet"}
    try:
        markets = await ex.load_markets()
        for sym in SYMBOLS:
            m = markets.get(sym)
            if not m:
                pf["symbols"][sym] = {"ok": False, "reason": "symbol_not_available"}
                pf["ok"] = False
            else:
                limits = m.get("limits") or {}
                pf["symbols"][sym] = {
                    "ok": bool(m.get("active", True)),
                    "base": m.get("base"),
                    "quote": m.get("quote"),
                    "min_amount": (limits.get("amount") or {}).get("min"),
                    "min_cost": (limits.get("cost") or {}).get("min"),
                }
        if BOT_MODE == "testnet" and EXECUTE_TESTNET_ORDERS and not (API_KEY and API_SECRET):
            pf["ok"] = False
            pf["auth_error"] = "execution_enabled_but_credentials_missing"
    except Exception as e:
        pf["ok"] = False
        pf["error"] = str(e)[:250]
    state["preflight"] = pf


async def loop():
    await run_preflight()
    state["status"] = "running" if state["preflight"].get("ok") else "preflight_warning"
    while True:
        try:
            day = datetime.now(timezone.utc).date().isoformat()
            if state["day_key"] != day:
                state["day_key"] = day
                state["day_start_equity"] = state["paper_balance"] if BOT_MODE == "paper" else max(float(state["day_start_equity"]), 1.0)
                state["realized_pnl"] = 0.0
            await manage_position()
            await scan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            state["status"] = "error:" + type(e).__name__
            state["errors"].append({"time": now_iso(), "error": str(e)[:220]})
            state["errors"] = state["errors"][-30:]
        await asyncio.sleep(LOOP_SECONDS)


@app.on_event("startup")
async def startup():
    global task
    task = asyncio.create_task(loop())


@app.on_event("shutdown")
async def shutdown():
    if task:
        task.cancel()
    await ex.close()


@app.get("/health")
async def health():
    return {"ok": True, "version": state["version"], "mode": BOT_MODE, "status": state["status"], "execution_enabled": state["execution_enabled"]}


@app.get("/preflight")
async def preflight():
    return state["preflight"]


@app.get("/status")
async def status():
    public = dict(state)
    public["config"] = {
        "symbols": SYMBOLS,
        "entry_score": ENTRY_SCORE,
        "max_fraction": MAX_FRACTION,
        "max_quote_per_trade": MAX_QUOTE_PER_TRADE,
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "stop_atr": STOP_ATR,
        "take_profit_atr": TP_ATR,
        "trailing_atr": TRAIL_ATR,
        "cooldown_minutes": COOLDOWN_MINUTES,
    }
    return public


@app.get("/", response_class=HTMLResponse)
async def home():
    p = state["position"]
    sig = state["last_signal"]
    trades = state["trades"][-10:]
    st = state["stats"]
    mode_label = "PAPER — no real orders" if BOT_MODE == "paper" else ("BINANCE SPOT TESTNET — execution ON" if EXECUTE_TESTNET_ORDERS else "BINANCE SPOT TESTNET — signal only")
    return f"""<!doctype html>
<html><head><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Khafi Spot Bot v3</title>
<style>
body{{font-family:system-ui;max-width:820px;margin:24px;background:#fafafa;color:#171717}}
.card{{padding:16px;border:1px solid #ddd;border-radius:16px;background:white;margin:12px 0}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}
.good{{font-weight:700}} .warn{{font-weight:700}}
</style></head><body>
<h1>Khafi Spot Bot v3</h1>
<div class='card'><b>Mode:</b> {mode_label}<br><b>Status:</b> {state['status']}<br><b>Symbols:</b> {', '.join(SYMBOLS)}<br><b>Preflight:</b> {state['preflight'].get('ok')}</div>
<div class='card'><b>Paper balance:</b> {state['paper_balance']:.2f}<br><b>Realized PnL:</b> {state['realized_pnl']:.4f}<br><b>Closed:</b> {st['closed']} &nbsp; <b>Wins:</b> {st['wins']} &nbsp; <b>Losses:</b> {st['losses']}</div>
<div class='card'><h3>Open position</h3><pre>{p}</pre></div>
<div class='card'><h3>Last signal</h3><pre>{sig}</pre></div>
<div class='card'><h3>Last trades</h3><pre>{trades}</pre></div>
<div class='card'><b>Safety lock:</b> this code has no Binance mainnet/live trading mode. Test execution is restricted to Binance Spot Testnet.</div>
</body></html>"""
