# -*- coding: utf-8 -*-
"""
幣安「合約測試網」下單機器人 — 組合策略 (A多空 50% + SMA50趨勢 50%)
=====================================================================
* 假錢。網址寫死 testnet, 程式會檢查, 不可能打到正式站。
* 金鑰只從環境變數讀:  BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET
* 預設 DRY_RUN=true -> 只印出「會下什麼單」, 不真的下單。
  要真的下單: 環境變數設 DRY_RUN=false
* 只在「日K完全收盤」後動作, 每根K只調倉一次 (bot_state.json 去重)

兩個引擎在同一帳戶【淨額】下單:
   目標曝險[幣] = 0.5*(A倉位/8) + 0.5*(趨勢倉位/3)
"""
import os, time, json, hmac, hashlib, warnings, csv
import numpy as np, pandas as pd, requests
from urllib.parse import urlencode
warnings.filterwarnings('ignore')

BASE = 'https://testnet.binancefuture.com'      # 【測試網,寫死】
assert 'testnet' in BASE, "安全檢查:只允許測試網"

KEY    = os.environ.get('BINANCE_TESTNET_KEY','')
SECRET = os.environ.get('BINANCE_TESTNET_SECRET','')
DRY_RUN = os.environ.get('DRY_RUN','true').lower() != 'false'

START_DATE = '2026-07-09'
WA, WT = 0.5, 0.5
A_COINS = ['BTC','ETH','SOL','LTC','LINK','ADA','DOGE','XLM']
T_COINS = ['BTC','ETH','SOL']
MIN_NOTIONAL_DELTA = 12.0    # 差額小於此就不下單(避免碎單/手續費)

HERE=os.path.dirname(os.path.abspath(__file__))
STATE=os.path.join(HERE,'bot_state.json')
LOG=os.path.join(HERE,'bot_log.csv')

# ---------------- 幣安 API ----------------
def signed(method, path, params=None):
    if not KEY or not SECRET: raise RuntimeError("缺少 BINANCE_TESTNET_KEY / SECRET 環境變數")
    p = dict(params or {}); p['timestamp']=int(time.time()*1000); p['recvWindow']=5000
    q = urlencode(p)
    sig = hmac.new(SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}{path}?{q}&signature={sig}"
    r = requests.request(method, url, headers={'X-MBX-APIKEY':KEY}, timeout=20)
    if r.status_code!=200: raise RuntimeError(f"{path} -> {r.status_code} {r.text[:200]}")
    return r.json()

def public(path, params=None):
    r=requests.get(BASE+path, params=params or {}, timeout=20)
    r.raise_for_status(); return r.json()

# ---------------- 訊號(與 paper_account 完全一致) ----------------
BN_HOSTS=['https://data-api.binance.vision','https://api.binance.com']
def fetch_bn(sym,days=400):
    end=int(time.time()*1000); start=end-days*86400*1000; rows=[]; cur=start
    while cur<end:
        r=None
        for h in BN_HOSTS:
            try:
                resp=requests.get(h+'/api/v3/klines',params=dict(symbol=sym,interval='1d',startTime=cur,limit=1000),timeout=20).json()
                if isinstance(resp,list): r=resp; break
            except: continue
        if not isinstance(r,list) or not r: break
        rows+=r; cur=r[-1][0]+1
        if len(r)<1000: break
    if not rows: return None
    df=pd.DataFrame(rows,columns=list('tohlcv')+['ct','qv','n','tb','tq','ig'])
    df['t']=pd.to_datetime(df['t'],unit='ms'); df['c']=df['c'].astype(float)
    return df.drop_duplicates('t').set_index('t')['c']

def fetch_cb(prod,days=400):
    end=int(time.time()); start=end-days*86400; rows=[]; cur=start
    while cur<end:
        seg=min(cur+250*86400,end)
        try:
            r=requests.get(f'https://api.exchange.coinbase.com/products/{prod}/candles',
              params=dict(granularity=86400,start=pd.to_datetime(cur,unit='s').isoformat(),
                          end=pd.to_datetime(seg,unit='s').isoformat()),timeout=20).json()
        except: r=None
        if isinstance(r,list): rows+=r
        cur=seg; time.sleep(0.1)
    if not rows: return None
    df=pd.DataFrame(rows,columns=['t','l','h','o','c','v']); df['t']=pd.to_datetime(df['t'],unit='s')
    return df.drop_duplicates('t').sort_values('t').set_index('t')['c'].astype(float)

def pos_A(prem,fng):
    cbL=prem>0; g=fng>60; p=pd.Series(0.0,index=prem.index)
    p[cbL&g]=1.0; p[(~cbL)&g]=-0.5; p[cbL&(~g)]=0.5; p[(~cbL)&(~g)]=-1.0
    return p

r=requests.get('https://api.alternative.me/fng/?limit=0&format=json',timeout=20).json()
fng=pd.DataFrame(r['data']); fng['value']=fng['value'].astype(int)
fng['date']=pd.to_datetime(fng['timestamp'].astype(int),unit='s').dt.normalize()
fng=fng.sort_values('date').set_index('date')['value']

Apos={}; Tpos={}
for c in A_COINS:
    bn=fetch_bn(c+'USDT'); cb=fetch_cb(c+'-USD')
    if bn is None or cb is None: print(f"  A {c} 資料失敗"); continue
    d=pd.DataFrame({'p':bn}).dropna()
    d['prem']=((cb.reindex(d.index,method='ffill')/d['p']-1)*10000).rolling(7).mean()
    d['fng']=fng.reindex(d.index.normalize()).fillna(50); d=d.dropna()
    Apos[c]=pos_A(d['prem'],d['fng'])
for c in T_COINS:
    px=fetch_bn(c+'USDT')
    if px is None: continue
    ma=px.rolling(50).mean(); p=(px>ma).astype(float); p.iloc[:50]=0
    Tpos[c]=p
if not Apos or not Tpos: print("資料失敗"); raise SystemExit

AP=pd.DataFrame(Apos); TP=pd.DataFrame(Tpos)
dates=AP.index.intersection(TP.index)
dates=dates[dates>=pd.Timestamp(START_DATE)]
today=pd.Timestamp.utcnow().tz_localize(None).normalize()
dates=dates[dates<today]                       # 只用已完全收盤的日K
if len(dates)==0: print("還沒有已收盤的日K"); raise SystemExit
D=dates[-1]

st = json.load(open(STATE,encoding='utf-8')) if os.path.exists(STATE) else {'last':None}
if st.get('last')==str(D.date()):
    print(f"日K {D.date()} 已調倉過, 略過 (正常去重)"); raise SystemExit

# ---------------- 計算淨額目標曝險 ----------------
nA,nT=len(AP.columns),len(TP.columns)
symbols=sorted(set(list(AP.columns)+list(TP.columns)))
w={}
for c in symbols:
    a = float(AP.loc[D,c])/nA if c in AP.columns else 0.0
    t = float(TP.loc[D,c])/nT if c in TP.columns else 0.0
    w[c] = WA*a + WT*t

print('='*66)
print(f"  幣安合約測試網 bot  |  日K {D.date()}  |  DRY_RUN={DRY_RUN}")
print('='*66)

# ---------------- 帳戶 & 交易所規則 ----------------
def _dec(step_str):
    """從 stepSize 字串('0.00100000')推出小數位數,避免浮點雜訊"""
    s=step_str.rstrip('0')
    return len(s.split('.')[1]) if '.' in s and s.split('.')[1] else 0
info=public('/fapi/v1/exchangeInfo')
filt={}
for s in info['symbols']:
    f={x['filterType']:x for x in s['filters']}
    ss=f['LOT_SIZE']['stepSize']
    filt[s['symbol']]=dict(step=float(ss), dec=_dec(ss),
                           minqty=float(f['LOT_SIZE']['minQty']),
                           minnot=float(f.get('MIN_NOTIONAL',{}).get('notional',5)))
acct=signed('GET','/fapi/v2/account')
equity=float(acct['totalMarginBalance'])
curpos={p['symbol']:float(p['positionAmt']) for p in acct['positions']}
print(f"\n  測試網權益: {equity:,.2f} USDT")

def rnd(q,step,dec):
    """無條件捨去到 step 的倍數, 並依 stepSize 的小數位數四捨五入掉浮點雜訊"""
    v=round(np.floor(round(abs(q)/step,8))*step, dec)
    return v*(1 if q>=0 else -1)

orders=[]
print(f"\n  {'幣':<6}{'目標曝險':>10}{'目標數量':>16}{'現有':>14}{'需下單':>18}")
for c in symbols:
    sym=c+'USDT'
    if sym not in filt: print(f"  {c}: 測試網無此合約, 跳過"); continue
    st_,dc = filt[sym]['step'], filt[sym]['dec']
    px=float(public('/fapi/v1/ticker/price',{'symbol':sym})['price'])
    tgt_qty = rnd(w[c]*equity/px, st_, dc)
    cur = curpos.get(sym,0.0)
    delta = tgt_qty-cur
    dnot = abs(delta)*px
    act = ''
    if dnot >= max(MIN_NOTIONAL_DELTA, filt[sym]['minnot']):
        side='BUY' if delta>0 else 'SELL'
        qty=rnd(abs(delta),st_,dc)
        if qty>=filt[sym]['minqty']:
            orders.append((sym,side,qty)); act=f"{side} {qty}"
    print(f"  {c:<6}{w[c]*100:>9.2f}%{tgt_qty:>16.6g}{cur:>14.4f}{act:>18}")

if not orders:
    print("\n  沒有需要調整的部位")
else:
    print(f"\n  共 {len(orders)} 筆單:")
    for sym,side,qty in orders:
        if DRY_RUN:
            print(f"    [DRY_RUN] 不下單 — 會下: {side} {qty} {sym}")
        else:
            try:
                res=signed('POST','/fapi/v1/order',{'symbol':sym,'side':side,'type':'MARKET','quantity':qty})
                print(f"    ✅ 成交 {side} {qty} {sym}  orderId={res.get('orderId')}")
            except Exception as e:
                print(f"    ❌ {sym} 下單失敗: {e}")

# ---------------- 記錄 ----------------
if not DRY_RUN:
    st['last']=str(D.date()); json.dump(st,open(STATE,'w'),ensure_ascii=False)
    newf=not os.path.exists(LOG)
    with open(LOG,'a',newline='',encoding='utf-8-sig') as f:
        wcsv=csv.writer(f)
        if newf: wcsv.writerow(['日K','執行UTC','權益USDT','下單數','淨曝險%'])
        wcsv.writerow([str(D.date()),pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M'),
                       round(equity,2),len(orders),round(sum(w.values())*100,2)])
    print(f"\n  ✅ 已記錄 (bot_log.csv)")
else:
    print(f"\n  (DRY_RUN 模式:未下單、未寫狀態。確認無誤後把 DRY_RUN 設成 false)")
print(f"\n  淨曝險合計: {sum(w.values())*100:+.1f}%   毛曝險: {sum(abs(v) for v in w.values())*100:.1f}%")
