from __future__ import annotations
import asyncio, os, math
from datetime import datetime, timezone
import ccxt.async_support as ccxt
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
COOLDOWN_MINUTES=int(os.getenv('COOLDOWN_MINUTES','45'))

app=FastAPI(title='Khafi Spot Bot Paper')
ex=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot'}})
state={
    'status':'starting','paper_balance':START_BAL,'realized_pnl':0.0,
    'day_start':START_BAL,'day_key':None,'position':None,'last_signal':None,
    'last_trade_at':None,'trades':[],'errors':[]
}
task=None

def now_iso(): return datetime.now(timezone.utc).isoformat()

def ema_at(values, span, idx):
    if not values or idx < 0: return 0.0
    alpha=2.0/(span+1.0); e=float(values[0])
    for i in range(1, min(idx+1,len(values))):
        e=alpha*float(values[i])+(1-alpha)*e
    return e

def rsi_at(values, period, idx):
    if idx < period: return 50.0
    gains=[]; losses=[]
    start=max(1,idx-period+1)
    for i in range(start,idx+1):
        d=float(values[i])-float(values[i-1])
        gains.append(max(d,0.0)); losses.append(max(-d,0.0))
    ag=sum(gains)/max(len(gains),1); al=sum(losses)/max(len(losses),1)
    if al==0: return 100.0 if ag>0 else 50.0
    rs=ag/al
    return 100.0-(100.0/(1.0+rs))

def atr_at(rows, period, idx):
    if idx < 1: return 0.0
    trs=[]
    start=max(1,idx-period+1)
    for i in range(start,idx+1):
        high=float(rows[i][2]); low=float(rows[i][3]); prev=float(rows[i-1][4])
        trs.append(max(high-low,abs(high-prev),abs(low-prev)))
    return sum(trs)/max(len(trs),1)

def vol_ratio_at(rows, period, idx):
    start=max(0,idx-period+1)
    vals=[float(rows[i][5]) for i in range(start,idx+1)]
    avg=sum(vals)/max(len(vals),1)
    return (float(rows[idx][5])/avg) if avg>0 else 0.0

async def ohlcv(symbol,tf):
    return await ex.fetch_ohlcv(symbol,timeframe=tf,limit=250)

def analyze(symbol,d15,d1h):
    if len(d15)<220 or len(d1h)<220:
        return {'symbol':symbol,'score':0,'reasons':['not_enough_data']}
    i15=len(d15)-2; i1h=len(d1h)-2
    c15=[float(r[4]) for r in d15]; c1h=[float(r[4]) for r in d1h]
    price=c15[i15]
    e20=ema_at(c15,20,i15); e50=ema_at(c15,50,i15)
    h50=ema_at(c1h,50,i1h); h200=ema_at(c1h,200,i1h)
    hprice=c1h[i1h]
    rv=rsi_at(c15,14,i15); av=atr_at(d15,14,i15); vr=vol_ratio_at(d15,20,i15)
    score=0; reasons=[]
    if h50>h200 and hprice>h50:
        score+=30; reasons.append('1h_uptrend')
    else: reasons.append('1h_not_uptrend')
    if e20>e50:
        score+=20; reasons.append('15m_trend')
    if price>e20:
        score+=10; reasons.append('above_ema20')
    if 47<=rv<=64:
        score+=15; reasons.append(f'rsi_ok:{rv:.1f}')
    elif 42<=rv<47:
        score+=8; reasons.append(f'rsi_early:{rv:.1f}')
    elif rv>70:
        score-=15; reasons.append(f'overbought:{rv:.1f}')
    if vr>=1.10:
        score+=15; reasons.append(f'volume_confirmed:{vr:.2f}')
    elif vr>=0.85:
        score+=8; reasons.append(f'volume_neutral:{vr:.2f}')
    else: reasons.append(f'volume_weak:{vr:.2f}')
    if av>0:
        ext=abs(price-e20)/av
        if ext<=0.8:
            score+=10; reasons.append(f'not_extended:{ext:.2f}ATR')
        elif ext>1.8:
            score-=10; reasons.append(f'too_extended:{ext:.2f}ATR')
    score=max(0,min(100,score))
    return {
        'time':now_iso(),'symbol':symbol,'score':score,'price':price,'atr':av,
        'rsi':round(rv,2),'volume_ratio':round(vr,2),'reasons':reasons,
        'stop':price-STOP_ATR*av,'tp':price+TP_ATR*av
    }

def cooldown_blocked():
    if not state['last_trade_at']: return False
    try:
        t=datetime.fromisoformat(state['last_trade_at'])
        return (datetime.now(timezone.utc)-t).total_seconds()/60.0 < COOLDOWN_MINUTES
    except Exception: return False

async def enter(sig):
    alloc=state['paper_balance']*MAX_FRACTION
    if alloc<=0 or sig.get('price',0)<=0 or sig.get('atr',0)<=0: return
    base=alloc/sig['price']
    state['paper_balance']-=alloc
    state['position']={
        'symbol':sig['symbol'],'entry':sig['price'],'base':base,'spent':alloc,
        'atr':sig['atr'],'stop':sig['stop'],'tp':sig['tp'],'highest':sig['price'],
        'opened':now_iso(),'trail':False
    }
    state['last_trade_at']=now_iso()
    state['trades'].append({'time':now_iso(),'type':'PAPER_BUY','symbol':sig['symbol'],'price':sig['price'],'quote':alloc,'score':sig['score'],'reasons':sig['reasons']})

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
        proceeds=p['base']*price; pnl=proceeds-p['spent']
        state['paper_balance']+=proceeds; state['realized_pnl']+=pnl
        state['trades'].append({'time':now_iso(),'type':'EXIT','symbol':p['symbol'],'price':price,'pnl':pnl,'reason':reason})
        state['last_trade_at']=now_iso(); state['position']=None

async def scan():
    if state['position']: return
    if cooldown_blocked():
        state['status']='cooldown'; return
    loss=max(0,-state['realized_pnl'])/max(state['day_start'],1e-9)*100
    if loss>=MAX_DAILY_LOSS_PCT:
        state['status']='daily_loss_lock'; return
    best=None
    for sym in SYMBOLS:
        try:
            d15,d1h=await asyncio.gather(ohlcv(sym,'15m'),ohlcv(sym,'1h'))
            s=analyze(sym,d15,d1h)
            if best is None or s.get('score',0)>best.get('score',0): best=s
        except Exception as e:
            state['errors'].append({'time':now_iso(),'symbol':sym,'error':str(e)[:180]})
            state['errors']=state['errors'][-20:]
    if best:
        state['last_signal']=best; state['status']=f"best:{best['symbol']}:{best['score']}"
        if best.get('score',0)>=ENTRY_SCORE: await enter(best)

async def loop():
    state['status']='running'
    while True:
        try:
            day=datetime.now(timezone.utc).date().isoformat()
            if state['day_key']!=day:
                state['day_key']=day; state['day_start']=state['paper_balance']; state['realized_pnl']=0.0
            await manage(); await scan()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            state['status']='error:'+type(e).__name__
            state['errors'].append({'time':now_iso(),'error':str(e)[:180]})
            state['errors']=state['errors'][-20:]
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
    return f"""<!doctype html><html><meta name='viewport' content='width=device-width,initial-scale=1'><body style='font-family:system-ui;max-width:760px;margin:24px'><h1>Khafi Spot Bot</h1><div style='padding:14px;border:1px solid #ddd;border-radius:14px'><b>Mode:</b> PAPER ONLY<br><b>Status:</b> {state['status']}<br><b>Balance:</b> {state['paper_balance']:.4f} USDC<br><b>Realized PnL:</b> {state['realized_pnl']:.4f} USDC</div><h3>Position</h3><pre style='white-space:pre-wrap'>{p}</pre><h3>Last signal</h3><pre style='white-space:pre-wrap'>{sig}</pre><h3>Last trades</h3><pre style='white-space:pre-wrap'>{trades}</pre><p><b>Safety lock:</b> this deployment cannot place real orders.</p></body></html>"""
