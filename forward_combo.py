# -*- coding: utf-8 -*-
"""
每日前向測試 — 組合策略 (A多空[CB溢價xFNG] + SMA50趨勢, 50/50) — 雲端版
================================================================
它會:
 1. 抓最新資料, 算今天的組合倉位 (A多空 8幣 + 趨勢 3幣)
 2. 算模擬金累積損益 (從 FORWARD_START, 本金 $10,000): 純A / 純趨勢 / 50/50組合 三條並列
 3. 算兩個引擎各自的 90日命中率 + 紅綠燈健康度
 4. 算 Sharpe 標準誤差 (回測 & forward), 告訴你這個 Sharpe 估計值可不可信
 5. 用日K日期去重寫進 combo_forward_log.csv (一天一筆, 多跑容錯)
一天跑幾次都沒關係。FORWARD_START 之後才是真驗證。
"""
import warnings, csv, os, numpy as np, pandas as pd, requests, time
warnings.filterwarnings('ignore')

FORWARD_START = '2026-07-09'          # 新組合策略的前向測試起始日
CAPITAL = 10000
WA = 0.5                               # A的資金權重 (趨勢=1-WA)
A_COINS = ['BTC','ETH','SOL','LTC','LINK','ADA','DOGE','XLM']   # 成熟流動幣 (edge有此邊界)
T_COINS = ['BTC','ETH','SOL']          # 趨勢引擎
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combo_forward_log.csv')

BN_HOSTS = ['https://data-api.binance.vision', 'https://api.binance.com']   # 鏡像優先(不擋美國IP)
def fetch_bn(sym, days=1200):
    end=int(time.time()*1000); start=end-days*86400*1000; rows=[]; cur=start
    while cur < end:
        r=None
        for host in BN_HOSTS:
            try:
                resp=requests.get(host+'/api/v3/klines',
                    params=dict(symbol=sym,interval='1d',startTime=cur,limit=1000),timeout=20).json()
                if isinstance(resp,list): r=resp; break
            except: continue
        if not isinstance(r,list) or not r: break
        rows+=r; cur=r[-1][0]+1
        if len(r)<1000: break
    if not rows: return None
    df=pd.DataFrame(rows,columns=list('tohlcv')+['ct','qv','n','tb','tq','ig'])
    df['t']=pd.to_datetime(df['t'],unit='ms'); df['c']=df['c'].astype(float)
    return df.drop_duplicates('t').set_index('t')['c']

def fetch_cb(prod, days=1200):
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
    cbL=prem>0; greed=fng>60; p=pd.Series(0.0,index=prem.index)
    p[cbL&greed]=1.0; p[(~cbL)&greed]=-0.5; p[cbL&(~greed)]=0.5; p[(~cbL)&(~greed)]=-1.0
    return p

# --- FNG ---
r=requests.get('https://api.alternative.me/fng/?limit=0&format=json',timeout=20).json()
fng=pd.DataFrame(r['data']); fng['value']=fng['value'].astype(int)
fng['date']=pd.to_datetime(fng['timestamp'].astype(int),unit='s').dt.normalize()
fng=fng.sort_values('date').set_index('date')['value']

# --- 引擎A (多空) ---
A_rets={}; A_hit={}; A_sig=[]; candle=None
for c in A_COINS:
    bn=fetch_bn(c+'USDT'); cb=fetch_cb(c+'-USD')
    if bn is None or cb is None: print(f"  A {c} 資料失敗跳過"); continue
    d=pd.DataFrame({'p':bn}).dropna()
    d['prem']=((cb.reindex(d.index,method='ffill')/d['p']-1)*10000).rolling(7).mean()
    d['fng']=fng.reindex(d.index.normalize()).fillna(50); d=d.dropna()
    ret=d['p'].pct_change().fillna(0); pos=pos_A(d['prem'],d['fng']); pl=pos.shift(1).fillna(0)
    A_rets[c]=ret*pl - pl.diff().abs().fillna(0)*0.001
    A_hit[c]=(np.sign(pl)==np.sign(ret)).astype(float)
    A_sig.append((c,d['prem'].iloc[-1],d['fng'].iloc[-1],pos.iloc[-1]))
    candle=d.index[-1]

# --- 引擎B (SMA50趨勢, 多/現金) ---
T_rets={}; T_hit={}; T_sig=[]
for c in T_COINS:
    px=fetch_bn(c+'USDT')
    if px is None: print(f"  T {c} 資料失敗跳過"); continue
    ma=px.rolling(50).mean(); pos=(px>ma).astype(float); pos.iloc[:50]=0
    ret=px.pct_change().fillna(0); pl=pos.shift(1).fillna(0)
    T_rets[c]=ret*pl - pl.diff().abs().fillna(0)*0.0004
    nz=pl!=0
    T_hit[c]=(np.sign(pl)==np.sign(ret)).where(nz)
    T_sig.append((c,px.iloc[-1],ma.iloc[-1],pos.iloc[-1]))

if not A_rets or not T_rets:
    print("資料抓取失敗, 結束"); raise SystemExit

# --- 組合日報酬 ---
A=pd.DataFrame(A_rets); Aret=A.div((~A.isna()).sum(axis=1),axis=0).fillna(0).sum(axis=1)
T=pd.DataFrame(T_rets); Tret=T.mean(axis=1)
J=pd.concat([Aret.rename('A'),Tret.rename('T')],axis=1).dropna()
J['C']=WA*J['A']+(1-WA)*J['T']

# --- 模擬金損益 (從 FORWARD_START) ---
fs=pd.Timestamp(FORWARD_START)
def eq(series):
    s=series[series.index>=fs]; return (1+s).prod()
days_fwd=(J.index[-1]-fs).days if J.index[-1]>=fs else 0
eqA,eqT,eqC=eq(J['A']),eq(J['T']),eq(J['C'])

# --- 命中率 + 紅綠燈 ---
def light(h):
    return '🟢綠燈' if h>=52 else '🟡黃燈' if h>=48 else '🔴紅燈'
hitA=np.nanmean([A_hit[c].tail(90).mean()*100 for c in A_hit])
hitT=np.nanmean([T_hit[c].tail(90).mean()*100 for c in T_hit])

# --- Sharpe 標準誤差 ---
def sharpe(s,b=365):
    s=s.dropna(); return (s.mean()*b)/(s.std()*np.sqrt(b)) if s.std()>0 else 0
def se(SR,years):
    return np.sqrt((1+0.5*SR**2)/years) if years>0 else float('inf')
# 回測(全期)
SR_bt=sharpe(J['C']); yrs_bt=len(J)/365; se_bt=se(SR_bt,yrs_bt)
# forward(起始至今)
fwd=J['C'][J.index>=fs]; SR_fw=sharpe(fwd) if len(fwd)>2 else float('nan')
yrs_fw=days_fwd/365; se_fw=se(SR_fw if not np.isnan(SR_fw) else 1.0, yrs_fw)

# --- 印報告 ---
print('='*64)
print(f"  組合策略 (A多空 {int(WA*100)}% + SMA50趨勢 {int((1-WA)*100)}%)  |  日K: {candle.date()}")
print('='*64)
print(f"\n【引擎A 8幣訊號】溢價7d / FNG / 倉位")
for c,pr,fv,ps in A_sig:
    lsd='滿多' if ps==1 else '半多' if ps==0.5 else '半空' if ps==-0.5 else '滿空'
    print(f"  {c:5}{pr:>8.1f}bp{fv:>5.0f}{ps:>+6.1f} {lsd}")
print(f"\n【引擎B 趨勢訊號】價格 vs SMA50")
for c,px_,ma_,ps in T_sig:
    print(f"  {c:5} {'價>均→做多' if ps==1 else '價<均→現金':}  (px {px_:.2f} / ma {ma_:.2f})")

print(f"\n--- 模擬金 (本金 ${CAPITAL:,}, 起始 {FORWARD_START}, 已 {days_fwd} 天) ---")
print(f"  純A     : ${CAPITAL*eqA:,.0f}  ({(eqA-1)*100:+.1f}%)")
print(f"  純趨勢   : ${CAPITAL*eqT:,.0f}  ({(eqT-1)*100:+.1f}%)")
print(f"  50/50組合: ${CAPITAL*eqC:,.0f}  ({(eqC-1)*100:+.1f}%)   ← 主策略")

print(f"\n--- 健康度 (90日命中率) ---")
print(f"  引擎A(CB溢價): {hitA:.1f}%  {light(hitA)}")
print(f"  引擎B(趨勢)  : {hitT:.1f}%  {light(hitT)}")
print(f"  規則: ≥52%綠 / 48-52%黃 / 連續兩月<48%=紅=疑似該引擎regime失效→減碼")

print(f"\n--- Sharpe 可信度 (標準誤差 √((1+0.5·SR²)/年數)) ---")
print(f"  回測(此腳本近{yrs_bt:.1f}年資料,不含2021): Sharpe {SR_bt:.2f} ± {se_bt:.2f}  (95%區間約 [{SR_bt-2*se_bt:.1f}, {SR_bt+2*se_bt:.1f}])")
if yrs_fw>0.05 and not np.isnan(SR_fw):
    print(f"  Forward : Sharpe {SR_fw:.2f} ± {se_fw:.1f}  (才{yrs_fw:.2f}年 → 誤差巨大, 現在的數字沒意義)")
else:
    print(f"  Forward : 才 {days_fwd} 天, 樣本太少, Sharpe 還算不出有意義的值")
print(f"  ★ 提醒: 慢性衰退要~3年才勉強看得出, 別因forward短期好壞就加碼或砍策略")

# --- 寫 log (日K日期去重) ---
key=str(candle.date()); existing=set()
if os.path.exists(LOG):
    with open(LOG,encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row)>1: existing.add(row[0])
if key in existing:
    print(f"\n(日K {key} 已記錄, 略過寫入 — 正常去重)")
else:
    newfile=not os.path.exists(LOG)
    with open(LOG,'a',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        if newfile:
            w.writerow(['日K基準','執行UTC','組合$','純A$','純趨勢$','A命中率','趨勢命中率','A燈','趨勢燈','forward天數'])
        now=pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')
        w.writerow([key,now,round(CAPITAL*eqC),round(CAPITAL*eqA),round(CAPITAL*eqT),
                    round(hitA,1),round(hitT,1),light(hitA),light(hitT),days_fwd])
    print(f"\n✅ 已記錄日K {key} 到 combo_forward_log.csv")
