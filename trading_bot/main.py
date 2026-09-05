from __future__ import annotations

import asyncio
import os
import time
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
MAX_TRADES_PER_DAY = min(max(int(os.getenv("MAX_TRADES_PER_DAY", "6")), 1), 20)
STOP_ATR = min(max(float(os.getenv("STOP_ATR_MULT", "1.6")), 0.8), 3.0)
TP_ATR = min(max(float(os.getenv("TAKE_PROFIT_ATR_MULT", "2.8")), 1.2), 6.0)
TRAIL_ATR = min(max(float(os.getenv("TRAILING_ATR_MULT", "1.1")), 0.5), 3.0)
TRAIL_ACTIVATE = min(max(float(os.getenv("TRAILING_ACTIVATE_ATR", "1.2")), 0.5), 3.0)
MAX_HOLD_HOURS = max(1, int(os.getenv("MAX_HOLD_HOURS", "24")))

if BOT_MODE not in {"paper", "testnet"}:
    raise RuntimeError("Safety lock: BOT_MODE supports only 'paper' or 'testnet'. Live/mainnet is intentionally unavailable.")

app = FastAPI(title="Khafi Spot Bot v3.2")

exchange_args: dict[str, Any] = {
    "enableRateLimit": True,
    "options": {"defaultType": "spot"},
}
if BOT_MODE == "testnet" and API_KEY and API_SECRET:
    exchange_args.update({"apiKey": API_KEY, "secret": API_SECRET})

ex = ccxt.binance(exchange_args)
if BOT_MODE == "testnet":
    # Hard safety boundary: every Binance call goes to Spot Testnet in testnet mode.
    ex.set_sandbox_mode(True)

STARTED_AT = datetime.now(timezone.utc)

state: dict[str, Any] = {
    "version": "v3.2",
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
    "last_scan_at": None,
    "loop_count": 0,
    "trades": [],
    "errors": [],
    "stats": {"wins": 0, "losses": 0, "closed": 0, "gross_profit": 0.0, "gross_loss": 0.0},
    "risk": {"bot_buys_today": 0, "max_trades_per_day": MAX_TRADES_PER_DAY},
    "auth": {
        "configured": bool(API_KEY and API_SECRET),
        "checked": False,
        "ok": False,
        "read_only_check": True,
    },
    "recovery": {
        "checked": BOT_MODE != "testnet",
        "ok": BOT_MODE != "testnet",
        "safe_to_trade": BOT_MODE != "testnet",
        "bot_orders_found": 0,
        "recovered_position": False,
    },
    "preflight": {"ok": False, "symbols": {}, "auth_configured": bool(API_KEY and API_SECRET)},
}

task: asyncio.Task | None = None
_last_signal_log_key: str | None = None
_last_signal_log_at = 0.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_from_ms(ms: Any) -> str | None:
    try:
        if ms is None:
            return None
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def client_order_id(side: str) -> str:
    # Binance clientOrderId max length is comfortably above this short tag.
    stamp = int(time.time() * 1000)
    return f"khafi_{side}_{stamp}"


def order_client_id(order: dict[str, Any]) -> str:
    info = order.get("info") or {}
    return str(order.get("clientOrderId") or info.get("clientOrderId") or info.get("origClientOrderId") or "")


def is_khafi_order(order: dict[str, Any]) -> bool:
    return order_client_id(order).startswith("khafi_")


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
    gains: list[float] = []
    losses: list[float] = []
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
    trs: list[float] = []
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

    i15 = len(d15) - 2
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


def testnet_execution_allowed() -> bool:
    if BOT_MODE != "testnet":
        return False
    return bool(
        EXECUTE_TESTNET_ORDERS
        and state["auth"].get("ok")
        and state["recovery"].get("ok")
        and state["recovery"].get("safe_to_trade")
    )


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
    if not testnet_execution_allowed():
        state["status"] = "testnet_execution_safety_block"
        return
    if int(state["risk"].get("bot_buys_today", 0)) >= MAX_TRADES_PER_DAY:
        state["status"] = "max_trades_per_day_lock"
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

    cid = client_order_id("b")
    order = await ex.create_market_buy_order(sig["symbol"], amount, {"newClientOrderId": cid})
    filled = float(order.get("filled") or amount)
    avg = float(order.get("average") or px)
    if filled <= 0:
        state["status"] = "testnet_buy_not_filled"
        return

    state["position"] = {
        "symbol": sig["symbol"], "entry": avg, "base": filled, "spent": filled * avg,
        "atr": sig["atr"], "stop": avg - STOP_ATR * sig["atr"], "tp": avg + TP_ATR * sig["atr"],
        "highest": avg, "opened": now_iso(), "trail": False, "venue": "binance_spot_testnet",
        "order_id": order.get("id"), "client_order_id": cid, "recovered": False,
    }
    state["last_trade_at"] = now_iso()
    state["risk"]["bot_buys_today"] = int(state["risk"].get("bot_buys_today", 0)) + 1
    state["trades"].append({
        "time": now_iso(), "type": "TESTNET_BUY", "symbol": sig["symbol"],
        "price": avg, "base": filled, "score": sig["score"], "client_order_id": cid,
    })


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
        if not testnet_execution_allowed():
            state["status"] = "testnet_exit_safety_block"
            return
        amount = float(ex.amount_to_precision(p["symbol"], p["base"]))
        if amount <= 0:
            state["status"] = "testnet_exit_bad_amount"
            return
        cid = client_order_id("s")
        order = await ex.create_market_sell_order(p["symbol"], amount, {"newClientOrderId": cid})
        fill_price = float(order.get("average") or price)
        filled = float(order.get("filled") or amount)
        if filled <= 0:
            state["status"] = "testnet_sell_not_filled"
            return
        price = fill_price

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

    state["trades"].append({
        "time": now_iso(), "type": "EXIT", "symbol": p["symbol"],
        "price": price, "pnl": pnl, "reason": reason, "venue": p.get("venue"),
    })
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


async def maybe_log_signal(best: dict[str, Any]):
    global _last_signal_log_key, _last_signal_log_at
    key = f"{best.get('symbol')}:{best.get('score')}:{best.get('eligible')}"
    now_mono = time.monotonic()
    if key != _last_signal_log_key or (now_mono - _last_signal_log_at) >= 900:
        safe = {
            "symbol": best.get("symbol"),
            "score": best.get("score"),
            "eligible": best.get("eligible"),
            "rsi": best.get("rsi"),
            "volume_ratio": best.get("volume_ratio"),
            "volatility_pct": best.get("volatility_pct"),
        }
        print(f"[khafi-signal] {safe}", flush=True)
        _last_signal_log_key = key
        _last_signal_log_at = now_mono


async def scan():
    state["last_scan_at"] = now_iso()
    if state["position"]:
        state["status"] = "managing_position"
        return
    if cooldown_blocked():
        state["status"] = "cooldown"
        return
    if daily_loss_pct() >= MAX_DAILY_LOSS_PCT:
        state["status"] = "daily_loss_lock"
        return
    if BOT_MODE == "testnet" and int(state["risk"].get("bot_buys_today", 0)) >= MAX_TRADES_PER_DAY:
        state["status"] = "max_trades_per_day_lock"
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
        await maybe_log_signal(best)
        if best.get("eligible"):
            await enter(best)


async def verify_testnet_auth() -> dict[str, Any]:
    result: dict[str, Any] = {
        "configured": bool(API_KEY and API_SECRET),
        "checked": False,
        "ok": False,
        "read_only_check": True,
        "sandbox": BOT_MODE == "testnet",
    }
    if BOT_MODE != "testnet":
        result.update({"checked": True, "ok": True, "reason": "not_testnet_mode"})
        return result
    if not (API_KEY and API_SECRET):
        result.update({"checked": True, "ok": False, "error": "credentials_missing"})
        return result

    try:
        # USER_DATA read only: never creates, cancels, buys or sells anything.
        bal = await ex.fetch_balance()
        total = bal.get("total") or {}
        nonzero_assets = sum(1 for v in total.values() if isinstance(v, (int, float)) and float(v) != 0)
        result.update({"checked": True, "ok": True, "nonzero_asset_count": nonzero_assets})
    except Exception as e:
        result.update({"checked": True, "ok": False, "error": f"{type(e).__name__}: {str(e)[:180]}"})
    return result


async def recover_testnet_state() -> dict[str, Any]:
    result: dict[str, Any] = {
        "checked": True,
        "ok": False,
        "safe_to_trade": False,
        "bot_orders_found": 0,
        "recovered_position": False,
    }
    if BOT_MODE != "testnet":
        result.update({"ok": True, "safe_to_trade": True, "reason": "not_testnet_mode"})
        return result
    if not state["auth"].get("ok"):
        result["error"] = "auth_not_verified"
        return result

    try:
        bot_orders: list[dict[str, Any]] = []
        for sym in SYMBOLS:
            try:
                orders = await ex.fetch_orders(sym, limit=100)
                for order in orders:
                    if is_khafi_order(order):
                        bot_orders.append(order)
            except Exception as e:
                result.setdefault("symbol_errors", {})[sym] = f"{type(e).__name__}: {str(e)[:120]}"

        bot_orders.sort(key=lambda o: int(o.get("timestamp") or 0))
        result["bot_orders_found"] = len(bot_orders)

        # Restore cooldown and daily trade count from durable Binance Testnet order history.
        if bot_orders:
            last_order = bot_orders[-1]
            state["last_trade_at"] = iso_from_ms(last_order.get("timestamp"))
            result["last_bot_order_at"] = state["last_trade_at"]

        day_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
        buys_today = 0
        for order in bot_orders:
            if (
                str(order.get("side") or "").lower() == "buy"
                and float(order.get("filled") or 0) > 0
                and float(order.get("timestamp") or 0) >= day_start
            ):
                buys_today += 1
        state["risk"]["bot_buys_today"] = buys_today

        filled_orders = [
            o for o in bot_orders
            if float(o.get("filled") or 0) > 0
            and str(o.get("status") or "").lower() in {"closed", "filled"}
        ]
        buys = [o for o in filled_orders if str(o.get("side") or "").lower() == "buy"]
        sells = [o for o in filled_orders if str(o.get("side") or "").lower() == "sell"]
        last_buy = max(buys, key=lambda o: int(o.get("timestamp") or 0), default=None)
        last_sell = max(sells, key=lambda o: int(o.get("timestamp") or 0), default=None)

        if last_buy and (
            not last_sell
            or int(last_buy.get("timestamp") or 0) > int(last_sell.get("timestamp") or 0)
        ):
            sym = str(last_buy.get("symbol") or "")
            if sym not in SYMBOLS:
                raise RuntimeError("Recovered order symbol is outside configured symbol allowlist.")

            filled = float(last_buy.get("filled") or 0)
            avg = float(last_buy.get("average") or 0)
            cost = float(last_buy.get("cost") or 0)
            if avg <= 0 and filled > 0 and cost > 0:
                avg = cost / filled
            if filled <= 0 or avg <= 0:
                raise RuntimeError("Could not reconstruct the latest Khafi buy fill.")

            d15 = await ohlcv(sym, "15m")
            i15 = max(1, len(d15) - 2)
            av = atr_at(d15, 14, i15)
            if av <= 0:
                ticker = await ex.fetch_ticker(sym)
                px = float(ticker.get("last") or avg)
                av = max(px * 0.005, 1e-9)

            ticker = await ex.fetch_ticker(sym)
            current = float(ticker.get("last") or avg)
            opened = iso_from_ms(last_buy.get("timestamp")) or now_iso()
            state["position"] = {
                "symbol": sym,
                "entry": avg,
                "base": filled,
                "spent": cost if cost > 0 else filled * avg,
                "atr": av,
                "stop": avg - STOP_ATR * av,
                "tp": avg + TP_ATR * av,
                "highest": max(avg, current),
                "opened": opened,
                "trail": current >= avg + TRAIL_ACTIVATE * av,
                "venue": "binance_spot_testnet",
                "order_id": last_buy.get("id"),
                "client_order_id": order_client_id(last_buy),
                "recovered": True,
            }
            result["recovered_position"] = True
            result["recovered_symbol"] = sym

        # If symbol reads failed entirely, do not allow new orders because recovery is uncertain.
        symbol_errors = result.get("symbol_errors") or {}
        if symbol_errors and len(symbol_errors) >= len(SYMBOLS):
            result["error"] = "could_not_read_any_symbol_order_history"
            return result

        result["ok"] = True
        result["safe_to_trade"] = True
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:220]}"
        return result


async def run_preflight():
    pf: dict[str, Any] = {
        "ok": True,
        "symbols": {},
        "auth_configured": bool(API_KEY and API_SECRET),
        "sandbox": BOT_MODE == "testnet",
        "execution_enabled": EXECUTE_TESTNET_ORDERS if BOT_MODE == "testnet" else False,
    }
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

        state["auth"] = await verify_testnet_auth()
        pf["auth_ok"] = state["auth"].get("ok")
        if BOT_MODE == "testnet" and not state["auth"].get("ok"):
            pf["ok"] = False
            pf["auth_error"] = state["auth"].get("error", "auth_failed")

        if BOT_MODE == "testnet" and state["auth"].get("ok"):
            state["recovery"] = await recover_testnet_state()
            pf["recovery_ok"] = state["recovery"].get("ok")
            pf["recovered_position"] = state["recovery"].get("recovered_position")
            if not state["recovery"].get("ok"):
                pf["ok"] = False
                pf["recovery_error"] = state["recovery"].get("error", "recovery_failed")
    except Exception as e:
        pf["ok"] = False
        pf["error"] = f"{type(e).__name__}: {str(e)[:220]}"

    state["preflight"] = pf
    print(f"[khafi-preflight] {pf}", flush=True)
    print(f"[khafi-auth] {state['auth']}", flush=True)
    print(f"[khafi-recovery] {state['recovery']}", flush=True)


async def loop():
    await run_preflight()
    state["status"] = "running" if state["preflight"].get("ok") else "preflight_warning"
    while True:
        try:
            state["loop_count"] += 1
            day = datetime.now(timezone.utc).date().isoformat()
            if state["day_key"] != day:
                state["day_key"] = day
                state["day_start_equity"] = state["paper_balance"] if BOT_MODE == "paper" else max(float(state["day_start_equity"]), 1.0)
                state["realized_pnl"] = 0.0
                if BOT_MODE != "testnet":
                    state["risk"]["bot_buys_today"] = 0
            await manage_position()
            await scan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            state["status"] = "error:" + type(e).__name__
            state["errors"].append({"time": now_iso(), "error": str(e)[:220]})
            state["errors"] = state["errors"][-30:]
            print(f"[khafi-error] {type(e).__name__}: {str(e)[:220]}", flush=True)
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
    uptime = int((datetime.now(timezone.utc) - STARTED_AT).total_seconds())
    return {
        "ok": True,
        "version": state["version"],
        "mode": BOT_MODE,
        "status": state["status"],
        "execution_enabled": state["execution_enabled"],
        "execution_allowed": testnet_execution_allowed() if BOT_MODE == "testnet" else False,
        "auth_ok": state["auth"].get("ok"),
        "recovery_ok": state["recovery"].get("ok"),
        "recovered_position": state["recovery"].get("recovered_position"),
        "uptime_seconds": uptime,
        "last_scan_at": state["last_scan_at"],
    }


@app.get("/auth-check")
async def auth_check():
    state["auth"] = await verify_testnet_auth()
    return state["auth"]


@app.get("/preflight")
async def preflight():
    return state["preflight"]


@app.get("/safety")
async def safety():
    return {
        "mode": BOT_MODE,
        "sandbox": BOT_MODE == "testnet",
        "execution_enabled": state["execution_enabled"],
        "execution_allowed": testnet_execution_allowed() if BOT_MODE == "testnet" else False,
        "auth": state["auth"],
        "recovery": state["recovery"],
        "risk": state["risk"],
        "mainnet_available": False,
    }


@app.get("/status")
async def status():
    public = dict(state)
    public["config"] = {
        "symbols": SYMBOLS,
        "entry_score": ENTRY_SCORE,
        "max_fraction": MAX_FRACTION,
        "max_quote_per_trade": MAX_QUOTE_PER_TRADE,
        "min_quote_per_trade": MIN_QUOTE_PER_TRADE,
        "max_daily_loss_pct": MAX_DAILY_LOSS_PCT,
        "max_trades_per_day": MAX_TRADES_PER_DAY,
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
    execution_allowed = testnet_execution_allowed() if BOT_MODE == "testnet" else False
    mode_label = (
        "PAPER — no exchange orders"
        if BOT_MODE == "paper"
        else ("BINANCE SPOT TESTNET — execution ON" if EXECUTE_TESTNET_ORDERS else "BINANCE SPOT TESTNET — signal only")
    )
    return f"""<!doctype html>
<html><head><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Khafi Spot Bot v3.2</title>
<style>
body{{font-family:system-ui;max-width:860px;margin:24px;background:#fafafa;color:#171717}}
.card{{padding:16px;border:1px solid #ddd;border-radius:16px;background:white;margin:12px 0}}
pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}
</style></head><body>
<h1>Khafi Spot Bot v3.2</h1>
<div class='card'>
<b>Mode:</b> {mode_label}<br>
<b>Status:</b> {state['status']}<br>
<b>Symbols:</b> {', '.join(SYMBOLS)}<br>
<b>Preflight:</b> {state['preflight'].get('ok')}<br>
<b>API auth verified:</b> {state['auth'].get('ok')}<br>
<b>Restart recovery:</b> {state['recovery'].get('ok')}<br>
<b>Execution safety gate:</b> {execution_allowed}
</div>
<div class='card'>
<b>Bot buys today:</b> {state['risk'].get('bot_buys_today')} / {MAX_TRADES_PER_DAY}<br>
<b>Realized PnL (current process):</b> {state['realized_pnl']:.4f}<br>
<b>Closed (current process):</b> {st['closed']} &nbsp; <b>Wins:</b> {st['wins']} &nbsp; <b>Losses:</b> {st['losses']}
</div>
<div class='card'><h3>Open position</h3><pre>{p}</pre></div>
<div class='card'><h3>Last signal</h3><pre>{sig}</pre></div>
<div class='card'><h3>Last trades in current process</h3><pre>{trades}</pre></div>
<div class='card'><b>Safety lock:</b> live/mainnet trading does not exist in this build. All executable exchange orders are restricted to Binance Spot Testnet.</div>
</body></html>"""
