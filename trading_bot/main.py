from __future__ import annotations
import asyncio, os, math
from datetime import datetime, timezone
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

SYMBOLS=[s.strip() for s in os.getenv('SYMBOLS','BTC/USDC,ETH/USDC,BNB/USDC').split(',') if s.strip()]
ENTRY_SCORE=float(os.getenv('ENTRY_SCORE','75'))
LOOP_SECONDS=int(os.getenv('LOOP_SECONDS','60'))
START_BAL=float(os.getenv('STARTING_PAPER_BALANCE','6'))
MAX_FRACTION=min(float(os.getenv('MAX_CAPITAL_FRACTION_PER_TRADE','0.20')),0.25)
MAX_DAILY_LOSS_PCT=min(float(os.getenv('MAX_DAILY_LOSS_PCT','2.0')),3.0)
STOP_ATR=float(os.getenv('STOP_ATR_MULT','1.5'))
TP_ATR=float(os.getenv('TAKE_PROFIT_ATR_MULT','2.5'))
TRAIL_ATR=float(os.getenv('TRAILING_ATR_MULT','1.0'))
TRAIL_ACTIVATE=float(os.getenv('TRAILING_ACTIVATE_ATR','1.0'))
MAX_HOLD_HOURS=int(os.getenv('MAX_HOLD_HOURS','24'))

app=FastAPI(title='Khafi Spot Bot Paper')
ex=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot'}})
state={'status':'starting','paper_balance':START_BAL,'realized_pnl':0.0,'day_start':START_BAL,'day_key':None,'position':None,'last_signal':None,'trades':[]}
task=None


def ema(s,span): return s.ewm(span=span,adjust=False).mean()
def rsi(close,period=14):
    d=close.diff(); up=d.clip(lower=0); down=-d.clip(upper=0)
    ag=up.ewm(alpha=1/period,adjust=False).mean(); al=down.ewm(alpha=1/period,adjust=False).mean()
    rs=ag/al.replace(0,np.nan); return (100-(100/(1+rs))).fillna(50)
def atr(df,period=14):
    pc=df['close'].shift(1)
    tr=pd.concat([df['high']-df['low'],(df['high']-pc).abs(),(df['low']-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,adjust=False).mean()
def enrich(df):
    o=df.copy(); o['ema20']=ema(o.close,20); o['ema50']=ema(o.close,50); o['ema200']=ema(o.close,200)
    o['rsi']=rsi(o.close); o['atr']=atr(o); o['vma']=o.volume.rolling(20).mean(); o['vr']=o.volume/o.vma.replace(0,np.nan); return o

async def ohlcv(symbol,tf):
    rows=await ex.fetch_ohlcv(symbol,timeframe=tf,limit=250)
    return pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])

def analyze(symbol,d15,d1h):
    a=enrich(d15); h=enrich(d1h); x=a.iloc[-2]; hx=h.iloc[-2]
    price=float(x.close); av=float(x.atr); score=0; reasons=[]
    if hx.ema50>hx.ema200 and hx.close>hx.ema50: score+=30; reasons.append('1h_uptrend')
    else: reasons.append('1h_not_uptrend')
    if x.ema20>x.ema50: score+=20; reasons.append('15m_trend')
    if price>x.ema20: score+=10; reasons.append('above_ema20')
    rv=float(x.rsi); vr=float(x.vr) if math.isfinite(float(x.vr)) else 0
    if 47<=rv<=64: score+=15; reasons.append(f'rsi:{rv:.1f}')
    elif 42<=rv<47: score+=8; reasons.append(f'rsi_early:{rv:.1f}')
    elif rv>70: score-=15; reasons.append('overbought')
    if vr>=1.10: score+=15; reasons.append(f'volume:{vr:.2f}')
    elif vr>=0.85: score+=8; reasons.append(f'volume_neutral:{vr:.2f}')
    else: reasons.append(f'volume_weak:{vr:.2f}')
    if av>0:
        ext=abs(price-float(x.ema20))/av
        if ext<=0.8: score+=10; reasons.append(f'not_extended:{ext:.2f}ATR')
        elif ext>1.8: score-=10; reasons.append(f'too_extended:{ext:.2f}ATR')
    score=max(0,min(100,score))
    return {'symbol':symbol,'score':score,'price':price,'atr':av,'reasons':reasons,'stop':price-STOP_ATR*av,'tp':price+TP_ATR*av}

async def enter(sig):
    alloc=state['paper_balance']*MAX_FRACTION
    if alloc<=0: return
    base=alloc/sig['price']
    state['paper_balance']-=alloc
    state['position']={'symbol':sig['symbol'],'entry':sig['price'],'base':base,'spent':alloc,'atr':sig['atr'],'stop':sig['stop'],'tp':sig['tp'],'highest':sig['price'],'opened':datetime.now(timezone.utc).isoformat(),'trail':False}
    state['trades'].append({'time':datetime.now(timezone.utc).isoformat(),'type':'PAPER_BUY','symbol':sig['symbol'],'price':sig['price'],'quote':alloc,'score':sig['score'],'reasons':sig['reasons']})

async def manage():
    p=state['position']
    if not p: return
    t=await ex.fetch_ticker(p['symbol']); price=float(t['last']); p['highest']=max(p['highest'],price)
    if not p['trail'] and price>=p['entry']+TRAIL_ACTIVATE*p['atr']: p['trail']=True
    stop=p['stop']
    if p['trail']: stop=max(stop,p['highest']-TRAIL_ATR*p['atr'])
    held=(datetime.now(timezone.utc)-datetime.fromisoformat(p['opened'])).total_seconds()/3600
    reason=None
    if price<=stop: reason='stop_or_trailing'
    elif price>=p['tp']: reason='take_profit'
    elif held>=MAX_HOLD_HOURS: reason='max_hold'
    if reason:
        proceeds=p['base']*price; pnl=proceeds-p['spent']; state['paper_balance']+=proceeds; state['realized_pnl']+=pnl
        state['trades'].append({'time':datetime.now(timezone.utc).isoformat(),'type':'EXIT','symbol':p['symbol'],'price':price,'pnl':pnl,'reason':reason}); state['position']=None

async def loop():
    state['status']='running'
    while True:
        try:
            day=datetime.now(timezone.utc).date().isoformat()
            if state['day_key']!=day:
                state['day_key']=day; state['day_start']=state['paper_balance']; state['realized_pnl']=0.0
            await manage()
            if not state['position']:
                loss=max(0,-state['realized_pnl'])/max(state['day_start'],1e-9)*100
                if loss>=MAX_DAILY_LOSS_PCT: state['status']='daily_loss_lock'
                else:
                    best=None
                    for sym in SYMBOLS:
                        try:
                            d15,d1h=await asyncio.gather(ohlcv(sym,'15m'),ohlcv(sym,'1h'))
                            s=analyze(sym,d15,d1h)
                            if best is None or s['score']>best['score']: best=s
                        except Exception as e:
                            state['last_signal']={'symbol':sym,'error':str(e)[:120]}
                    if best:
                        state['last_signal']=best; state['status']=f"best:{best['symbol']}:{best['score']}"
                        if best['score']>=ENTRY_SCORE: await enter(best)
        except Exception as e:
            state['status']='error:'+type(e).__name__
        await asyncio.sleep(LOOP_SECONDS)

@app.on_event('startup')
async def startup():
    global task; task=asyncio.create_task(loop())

@app.on_event('shutdown')
async def shutdown():
    if task: task.cancel()
    await ex.close()

@app.get('/health')
async def health(): return {'ok':True,'mode':'paper','status':state['status']}
@app.get('/status')
async def status(): return state
@app.get('/',response_class=HTMLResponse)
async def home():
    p=state['position']; sig=state['last_signal']; trades=state['trades'][-10:]
    return f"""<html><meta name='viewport' content='width=device-width,initial-scale=1'><body style='font-family:system-ui;max-width:760px;margin:24px'><h1>Khafi Spot Bot</h1><p><b>Mode:</b> PAPER ONLY</p><p><b>Status:</b> {state['status']}</p><p><b>Balance:</b> {state['paper_balance']:.4f} USDC</p><p><b>Realized PnL:</b> {state['realized_pnl']:.4f} USDC</p><hr><h3>Position</h3><pre style='white-space:pre-wrap'>{p}</pre><h3>Last signal</h3><pre style='white-space:pre-wrap'>{sig}</pre><h3>Last trades</h3><pre style='white-space:pre-wrap'>{trades}</pre><p><b>Safety:</b> live trading is disabled in this deployment.</p></body></html>"""
