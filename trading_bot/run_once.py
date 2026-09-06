from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt

# GitHub Actions executor: Binance Spot Testnet only. Mainnet/live is intentionally unavailable.
BOT_MODE = os.getenv("BOT_MODE", "testnet").strip().lower()
EXECUTE_TESTNET_ORDERS = os.getenv("EXECUTE_TESTNET_ORDERS", "true").strip().lower() in {"1", "true", "yes", "on"}
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
STATE_FILE = Path(os.getenv("STATE_FILE", "bot_state.json"))

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT").split(",") if s.strip()]
ENTRY_SCORE = float(os.getenv("ENTRY_SCORE", "78"))
COOLDOWN_MINUTES = max(15, int(os.getenv("COOLDOWN_MINUTES", "45")))
MAX_FRACTION = min(max(float(os.getenv("MAX_CAPITAL_FRACTION_PER_TRADE", "0.20")), 0.01), 0.25)
MAX_QUOTE_PER_TRADE = max(5.0, float(os.getenv("MAX_QUOTE_PER_TRADE", "25")))
MIN_QUOTE_PER_TRADE = max(5.0, float(os.getenv("MIN_QUOTE_PER_TRADE", "5")))
RISK_CAPITAL = max(25.0, float(os.getenv("RISK_CAPITAL", "100")))
MAX_DAILY_LOSS_PCT = min(max(float(os.getenv("MAX_DAILY_LOSS_PCT", "2.0")), 0.5), 3.0)
STOP_ATR = min(max(float(os.getenv("STOP_ATR_MULT", "1.5")), 0.8), 3.0)
TP_ATR = min(max(float(os.getenv("TAKE_PROFIT_ATR_MULT", "2.5")), 1.2), 6.0)
TRAIL_ATR = min(max(float(os.getenv("TRAILING_ATR_MULT", "1.0")), 0.5), 3.0)
TRAIL_ACTIVATE = min(max(float(os.getenv("TRAILING_ACTIVATE_ATR", "1.0")), 0.5), 3.0)
MAX_HOLD_HOURS = max(1, int(os.getenv("MAX_HOLD_HOURS", "24")))

if BOT_MODE != "testnet":
    raise RuntimeError("Safety lock: GitHub Actions executor supports Binance Spot Testnet only.")
if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing Binance Spot Testnet API credentials.")

DEFAULT_STATE: dict[str, Any] = {
    "version": "gha-v1",
    "status": "starting",
    "day_key": None,
    "day_realized_pnl": 0.0,
    "realized_pnl_total": 0.0,
    "position": None,
    "last_trade_at": None,
    "last_scan_candle": None,
    "last_signal": None,
    "stats": {"wins": 0, "losses": 0, "closed": 0, "gross_profit": 0.0, "gross_loss": 0.0},
    "trades": [],
    "errors": [],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_state() -> dict[str, Any]:
    state = json.loads(json.dumps(DEFAULT_STATE))
    if STATE_FILE.exists():
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state.update(raw)
                stats = dict(DEFAULT_STATE["stats"])
                stats.update(raw.get("stats") or {})
                state["stats"] = stats
        except Exception as exc:
            raise RuntimeError(f"State file is unreadable: {type(exc).__name__}") from exc
    state["trades"] = list(state.get("trades") or [])[-100:]
    state["errors"] = list(state.get("errors") or [])[-50:]
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(STATE_FILE)


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


def analyze(symbol: str, d15: list[list[Any]], d1h: list[list[Any]]) -> dict[str, Any]:
    if len(d15) < 220 or len(d1h) < 220:
        return {
            "time": now_iso(),
            "symbol": symbol,
            "score": 0,
            "eligible": False,
            "candle_ms": None,
            "reasons": ["not_enough_data"],
        }

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
        "candle_ms": int(d15[i15][0]),
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


def cooldown_blocked(state: dict[str, Any]) -> bool:
    raw = state.get("last_trade_at")
    if not raw:
        return False
    try:
        t = datetime.fromisoformat(raw)
        return (datetime.now(timezone.utc) - t).total_seconds() / 60.0 < COOLDOWN_MINUTES
    except Exception:
        return False


def daily_loss_pct(state: dict[str, Any]) -> float:
    loss = max(0.0, -float(state.get("day_realized_pnl") or 0.0))
    return loss / RISK_CAPITAL * 100.0


def append_error(state: dict[str, Any], exc: Exception, where: str) -> None:
    state.setdefault("errors", []).append({
        "time": now_iso(),
        "where": where,
        "type": type(exc).__name__,
        "error": str(exc)[:220],
    })
    state["errors"] = state["errors"][-50:]


async def preflight(ex: ccxt.binance) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "sandbox": True,
        "auth_configured": True,
        "execution_enabled": EXECUTE_TESTNET_ORDERS,
        "symbols": {},
    }
    markets = await ex.load_markets()
    for symbol in SYMBOLS:
        market = markets.get(symbol)
        if not market:
            result["ok"] = False
            result["symbols"][symbol] = {"ok": False, "reason": "symbol_not_available"}
            continue
        limits = market.get("limits") or {}
        active = bool(market.get("active", True))
        result["symbols"][symbol] = {
            "ok": active,
            "base": market.get("base"),
            "quote": market.get("quote"),
            "min_amount": (limits.get("amount") or {}).get("min"),
            "min_cost": (limits.get("cost") or {}).get("min"),
        }
        if not active:
            result["ok"] = False
    await ex.fetch_balance()  # read-only auth verification
    result["auth_ok"] = True
    return result


async def enter_position(ex: ccxt.binance, state: dict[str, Any], sig: dict[str, Any]) -> None:
    if not EXECUTE_TESTNET_ORDERS:
        state["status"] = "signal_only_execution_disabled"
        return

    market = ex.market(sig["symbol"])
    quote = market["quote"]
    bal = await ex.fetch_balance()
    free_quote = float((bal.get("free") or {}).get(quote, 0) or 0)
    alloc = min(free_quote * MAX_FRACTION, MAX_QUOTE_PER_TRADE)

    limits = market.get("limits") or {}
    min_cost = float((limits.get("cost") or {}).get("min") or 0.0)
    required_quote = max(MIN_QUOTE_PER_TRADE, min_cost)
    if alloc < required_quote:
        state["status"] = f"quote_too_small:{alloc:.4f}"
        return

    ticker = await ex.fetch_ticker(sig["symbol"])
    px = float(ticker.get("last") or sig["price"])
    amount = float(ex.amount_to_precision(sig["symbol"], alloc / px))
    min_amount = float((limits.get("amount") or {}).get("min") or 0.0)
    if amount <= 0 or amount < min_amount or amount * px < required_quote:
        state["status"] = "order_below_exchange_minimum"
        return

    order = await ex.create_market_buy_order(sig["symbol"], amount)
    filled = float(order.get("filled") or amount)
    avg = float(order.get("average") or px)
    spent = float(order.get("cost") or (filled * avg))
    if filled <= 0:
        state["status"] = "buy_not_filled"
        return

    state["position"] = {
        "symbol": sig["symbol"],
        "entry": avg,
        "base": filled,
        "spent": spent,
        "atr": float(sig["atr"]),
        "stop": avg - STOP_ATR * float(sig["atr"]),
        "tp": avg + TP_ATR * float(sig["atr"]),
        "highest": avg,
        "opened": now_iso(),
        "trail": False,
        "venue": "binance_spot_testnet",
        "order_id": order.get("id"),
    }
    state["last_trade_at"] = now_iso()
    state.setdefault("trades", []).append({
        "time": now_iso(),
        "type": "TESTNET_BUY",
        "symbol": sig["symbol"],
        "price": avg,
        "base": filled,
        "quote": spent,
        "score": sig["score"],
    })
    state["trades"] = state["trades"][-100:]
    state["status"] = f"entered:{sig['symbol']}"


async def close_position(ex: ccxt.binance, state: dict[str, Any], observed_price: float, reason: str) -> None:
    p = state.get("position")
    if not p:
        return
    if not EXECUTE_TESTNET_ORDERS:
        state["status"] = "exit_blocked_execution_disabled"
        return

    market = ex.market(p["symbol"])
    base_asset = market["base"]
    bal = await ex.fetch_balance()
    free_base = float((bal.get("free") or {}).get(base_asset, 0) or 0)
    requested = min(float(p["base"]), free_base)
    amount = float(ex.amount_to_precision(p["symbol"], requested))
    if amount <= 0:
        raise RuntimeError("No sellable base balance for tracked position")

    order = await ex.create_market_sell_order(p["symbol"], amount)
    filled = float(order.get("filled") or amount)
    avg = float(order.get("average") or observed_price)
    proceeds = float(order.get("cost") or (filled * avg))
    if filled <= 0:
        raise RuntimeError("Sell order was not filled")

    tracked_base = max(float(p["base"]), 1e-12)
    spent_portion = float(p["spent"]) * min(1.0, filled / tracked_base)
    pnl = proceeds - spent_portion

    state["realized_pnl_total"] = float(state.get("realized_pnl_total") or 0.0) + pnl
    state["day_realized_pnl"] = float(state.get("day_realized_pnl") or 0.0) + pnl
    stats = state["stats"]
    stats["closed"] = int(stats.get("closed") or 0) + 1
    if pnl >= 0:
        stats["wins"] = int(stats.get("wins") or 0) + 1
        stats["gross_profit"] = float(stats.get("gross_profit") or 0.0) + pnl
    else:
        stats["losses"] = int(stats.get("losses") or 0) + 1
        stats["gross_loss"] = float(stats.get("gross_loss") or 0.0) + abs(pnl)

    state.setdefault("trades", []).append({
        "time": now_iso(),
        "type": "TESTNET_EXIT",
        "symbol": p["symbol"],
        "price": avg,
        "base": filled,
        "pnl": pnl,
        "reason": reason,
    })
    state["trades"] = state["trades"][-100:]
    state["last_trade_at"] = now_iso()
    state["position"] = None
    state["status"] = f"closed:{reason}"


async def manage_position(ex: ccxt.binance, state: dict[str, Any]) -> None:
    p = state.get("position")
    if not p:
        return
    ticker = await ex.fetch_ticker(p["symbol"])
    price = float(ticker["last"])
    p["highest"] = max(float(p["highest"]), price)

    if not p.get("trail") and price >= float(p["entry"]) + TRAIL_ACTIVATE * float(p["atr"]):
        p["trail"] = True

    dynamic_stop = float(p["stop"])
    if p.get("trail"):
        dynamic_stop = max(dynamic_stop, float(p["highest"]) - TRAIL_ATR * float(p["atr"]))

    held_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(p["opened"])).total_seconds() / 3600.0
    if price <= dynamic_stop:
        await close_position(ex, state, price, "stop_or_trailing")
    elif price >= float(p["tp"]):
        await close_position(ex, state, price, "take_profit")
    elif held_hours >= MAX_HOLD_HOURS:
        await close_position(ex, state, price, "max_hold")
    else:
        state["status"] = f"holding:{p['symbol']}"


async def scan(ex: ccxt.binance, state: dict[str, Any]) -> None:
    if state.get("position"):
        return
    if cooldown_blocked(state):
        state["status"] = "cooldown"
        return
    if daily_loss_pct(state) >= MAX_DAILY_LOSS_PCT:
        state["status"] = "daily_loss_lock"
        return

    best: dict[str, Any] | None = None
    for symbol in SYMBOLS:
        try:
            d15, d1h = await asyncio.gather(
                ex.fetch_ohlcv(symbol, timeframe="15m", limit=250),
                ex.fetch_ohlcv(symbol, timeframe="1h", limit=250),
            )
            sig = analyze(symbol, d15, d1h)
            if best is None or float(sig.get("score") or 0) > float(best.get("score") or 0):
                best = sig
        except Exception as exc:
            append_error(state, exc, f"scan:{symbol}")

    if not best:
        state["status"] = "no_signal_data"
        return

    state["last_signal"] = best
    candle = best.get("candle_ms")
    if candle is not None and state.get("last_scan_candle") == candle:
        state["status"] = f"waiting_next_15m:{best['symbol']}:{best['score']}"
        return

    state["last_scan_candle"] = candle
    state["status"] = f"scanned:{best['symbol']}:{best['score']}"
    if best.get("eligible"):
        await enter_position(ex, state, best)


async def run() -> int:
    state = load_state()
    exchange_args: dict[str, Any] = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
    ex = ccxt.binance(exchange_args)
    ex.set_sandbox_mode(True)
    exit_code = 0

    try:
        pf = await preflight(ex)
        state["preflight"] = pf
        if not pf.get("ok") or not pf.get("auth_ok"):
            raise RuntimeError("Preflight failed")

        today = datetime.now(timezone.utc).date().isoformat()
        if state.get("day_key") != today:
            state["day_key"] = today
            state["day_realized_pnl"] = 0.0

        await manage_position(ex, state)
        await scan(ex, state)
    except Exception as exc:
        append_error(state, exc, "run")
        state["status"] = f"error:{type(exc).__name__}"
        exit_code = 1
    finally:
        state["updated_at"] = now_iso()
        save_state(state)
        await ex.close()

    public_summary = {
        "status": state.get("status"),
        "position": state.get("position"),
        "last_signal": state.get("last_signal"),
        "closed": state.get("stats", {}).get("closed", 0),
        "wins": state.get("stats", {}).get("wins", 0),
        "losses": state.get("stats", {}).get("losses", 0),
        "realized_pnl_total": round(float(state.get("realized_pnl_total") or 0.0), 8),
        "day_realized_pnl": round(float(state.get("day_realized_pnl") or 0.0), 8),
        "execution_enabled": EXECUTE_TESTNET_ORDERS,
        "sandbox": True,
    }
    print("KHAFI_RESULT=" + json.dumps(public_summary, separators=(",", ":"), default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
