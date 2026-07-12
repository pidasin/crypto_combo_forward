# -*- coding: utf-8 -*-
"""
真模擬金帳戶 — 組合策略 (A多空 50% + SMA50趨勢 50%)
========================================================
跟舊腳本的根本差別:
  * 餘額/部位存在 paper_state.json, 只往前累加, 【永遠不重算】
  * 漏跑幾天 -> 下次補跑那幾根K, 期間真的卡在原倉位
  * 成交 = 收盤價 + 滑價 + 手續費
  * 一旦寫入, 過去的紀錄不會因為你改策略而改變 (防自己作弊)

一天跑幾次都沒關係: 只處理「還沒處理過的日K」。
"""
import warnings, json, csv, os, numpy as np, pandas as pd, requests, time
warnings.filterwarnings('ignore')

START_DATE   = '2026-07-09'     # 策略凍結日 = 前向起算日 (絕對不可往回調)
CAPITAL      = 10000.0
WA           = 0.5              # A的資金權重
A_COINS      = ['BTC','ETH','SOL','LTC','LINK','ADA','DOGE','XLM']
T_COINS      = ['BTC','ETH','SOL']
A_FEE, A_SLIP = 0.0005, 0.0005  # A: 手續費5bp + 滑價5bp = 10bp/單位換手
T_FEE, T_SLIP = 0.0002, 0.0002  # 趨勢: 4bp/單位換手
BAND = 0.0024                   # 無交易帶(占權益): 對齊 bot 的 MIN_NOTIONAL_DELTA 12U/5000U

HERE  = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE,'paper_state.json')
LOG   = os.path.join(HERE,'paper_log.csv')

BN_HOSTS=['https://data-api.binance.vision','https://api.binance.com']
def fetch_bn(sym,days=800):
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

def fetch_cb(prod,days=800):
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

# ---------- 抓資料, 算每日訊號 ----------
r=requests.get('https://api.alternative.me/fng/?limit=0&format=json',timeout=20).json()
fng=pd.DataFrame(r['data']); fng['value']=fng['value'].astype(int)
fng['date']=pd.to_datetime(fng['timestamp'].astype(int),unit='s').dt.normalize()
fng=fng.sort_values('date').set_index('date')['value']

Apos={}; Aret={}; Apx={}   # Apx: 原始收盤價序列, 給「即時預覽」當基準價用
for c in A_COINS:
    bn=fetch_bn(c+'USDT'); cb=fetch_cb(c+'-USD')
    if bn is None or cb is None: print(f"  A {c} 資料失敗"); continue
    Apx[c]=bn
    d=pd.DataFrame({'p':bn}).dropna()
    d['prem']=((cb.reindex(d.index,method='ffill')/d['p']-1)*10000).rolling(7).mean()
    d['fng']=fng.reindex(d.index.normalize()).fillna(50); d=d.dropna()
    Apos[c]=pos_A(d['prem'],d['fng']); Aret[c]=d['p'].pct_change().fillna(0)
Tpos={}; Tret={}; Tpx={}
for c in T_COINS:
    px=fetch_bn(c+'USDT')
    if px is None: print(f"  T {c} 資料失敗"); continue
    Tpx[c]=px
    ma=px.rolling(50).mean(); p=(px>ma).astype(float); p.iloc[:50]=0
    Tpos[c]=p; Tret[c]=px.pct_change().fillna(0)
if not Apos or not Tpos: print("資料失敗, 結束"); raise SystemExit

AP=pd.DataFrame(Apos); AR=pd.DataFrame(Aret); TP=pd.DataFrame(Tpos); TR=pd.DataFrame(Tret)
dates=AP.index.intersection(TP.index)
dates=dates[dates>=pd.Timestamp(START_DATE)]
# 【關鍵】只處理「已完全收盤」的日K。今天那根還在形成中(close=當下即時價),
# 若處理它, 半天的報酬會被永久寫死進帳本 -> 必須排除, 等它明天收完再處理。
today_utc=pd.Timestamp.utcnow().tz_localize(None).normalize()
dates=dates[dates<today_utc]
if len(dates)==0:
    print(f"還沒有已收盤的日K可處理 (起算 {START_DATE}, 現在UTC {today_utc.date()})"); raise SystemExit

# ---------- 讀 state ----------
if os.path.exists(STATE):
    st=json.load(open(STATE,encoding='utf-8'))
else:
    st=dict(start=START_DATE,last=None,equity=CAPITAL,eqA=CAPITAL*WA,eqT=CAPITAL*(1-WA),
            wA={c:0.0 for c in AP.columns}, wT={c:0.0 for c in TP.columns}, ndays=0)

last=pd.Timestamp(st['last']) if st['last'] else None
has_pos = st['last'] is not None      # 已有部位? (第一根K之前沒有)
todo=[d for d in dates if last is None or d>last]
if not todo:
    print(f"沒有新的日K要處理 (最後處理: {st['last']}) — 正常去重")
else:
    print(f"要處理 {len(todo)} 根新日K: {todo[0].date()} ~ {todo[-1].date()}")

for d in todo:
    # 1) 先用「昨天設定的部位」吃今天的漲跌 (mark to market)
    if has_pos:
        rA=sum(st['wA'].get(c,0.0)*AR.loc[d,c] for c in AP.columns if not np.isnan(AR.loc[d,c]))
        rT=sum(st['wT'].get(c,0.0)*TR.loc[d,c] for c in TP.columns if not np.isnan(TR.loc[d,c]))
        st['eqA']*= (1+rA); st['eqT']*= (1+rT)
        # 1b) 【關鍵】權重會因價格漲跌而「漂移」, 不會自己留在原本的目標上。
        #     舊版忘了這步 -> 低估換手與成本。bot 是拿當下權益重算目標, 這裡要對齊。
        if (1+rA)!=0:
            st['wA']={c: st['wA'].get(c,0.0)*(1+AR.loc[d,c])/(1+rA) for c in AP.columns}
        if (1+rT)!=0:
            st['wT']={c: st['wT'].get(c,0.0)*(1+TR.loc[d,c])/(1+rT) for c in TP.columns}
    # 2) 收盤看到今天訊號 -> 從「漂移後的權重」調倉到新目標 (付手續費+滑價)
    #    BAND = 無交易帶, 對齊 bot 的 MIN_NOTIONAL_DELTA(差額太小就不下單)
    nA=len(AP.columns); nT=len(TP.columns)
    tgtA={c: float(AP.loc[d,c])/nA for c in AP.columns}
    tgtT={c: float(TP.loc[d,c])/nT for c in TP.columns}
    dA={c: (tgtA[c]-st['wA'].get(c,0.0)) for c in AP.columns}
    dT={c: (tgtT[c]-st['wT'].get(c,0.0)) for c in TP.columns}
    dA={c:(v if abs(v)>BAND else 0.0) for c,v in dA.items()}
    dT={c:(v if abs(v)>BAND else 0.0) for c,v in dT.items()}
    turnA=sum(abs(v) for v in dA.values()); turnT=sum(abs(v) for v in dT.values())
    st['eqA']-= st['eqA']*turnA*(A_FEE+A_SLIP)
    st['eqT']-= st['eqT']*turnT*(T_FEE+T_SLIP)
    st['wA']={c: st['wA'].get(c,0.0)+dA[c] for c in AP.columns}
    st['wT']={c: st['wT'].get(c,0.0)+dT[c] for c in TP.columns}
    # 3) 每日把兩腿資金再平衡回 50/50 (與回測一致)
    tot=st['eqA']+st['eqT']
    st['eqA']=tot*WA; st['eqT']=tot*(1-WA)
    st['equity']=tot; st['last']=str(d.date()); st['ndays']+=1; has_pos=True
    # 4) 寫 log (append-only, 永不改寫)
    newf=not os.path.exists(LOG)
    with open(LOG,'a',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        if newf: w.writerow(['日K','總權益$','A腿$','趨勢腿$','總報酬%','A淨曝險','趨勢投入','當日換手A','當日換手T'])
        w.writerow([str(d.date()),round(tot,2),round(st['eqA'],2),round(st['eqT'],2),
                    round((tot/CAPITAL-1)*100,2),round(sum(tgtA.values()),3),round(sum(tgtT.values()),3),
                    round(turnA,3),round(turnT,3)])

json.dump(st,open(STATE,'w',encoding='utf-8'),ensure_ascii=False,indent=1)

# ---------- 報告 ----------
def light(h): return '🟢綠燈' if h>=52 else '🟡黃燈' if h>=48 else '🔴紅燈'
hitA=np.nanmean([ (np.sign(AP[c].shift(1))==np.sign(AR[c])).astype(float).tail(90).mean()*100 for c in AP.columns])
hT=[]
for c in TP.columns:
    pl=TP[c].shift(1); nz=pl!=0
    hT.append((np.sign(pl)==np.sign(TR[c])).where(nz).tail(90).mean()*100)
hitT=np.nanmean(hT)
d=dates[-1]
# 【即時預覽用】記錄「最後鎖定那天」的收盤價當基準 — 手機端會拿現在的即時價
# 對照這個基準價 + 上次鎖定的倉位權重, 現場算出「如果現在收盤大概是多少」。
# 這個估計值不會寫進帳本, 純粹顯示用。
refA={c: float(Apx[c].loc[d]) for c in AP.columns if c in Apx and d in Apx[c].index}
refT={c: float(Tpx[c].loc[d]) for c in TP.columns if c in Tpx and d in Tpx[c].index}
print('='*58)
print(f"  真模擬金帳戶  |  日K {d.date()}  |  起算 {st['start']}  ({st['ndays']}個交易日)")
print('='*58)
print(f"\n  總權益: ${st['equity']:,.2f}   ({(st['equity']/CAPITAL-1)*100:+.2f}%)")
print(f"    A腿  : ${st['eqA']:,.2f}    淨曝險 {sum(st['wA'].values())*100:+.0f}%")
print(f"    趨勢腿: ${st['eqT']:,.2f}    投入 {sum(st['wT'].values())*100:.0f}%")
print(f"\n  今日目標倉位:")
for c in AP.columns: print(f"    A {c:5} {AP.loc[d,c]:+.1f}")
for c in TP.columns: print(f"    T {c:5} {'做多' if TP.loc[d,c]>0 else '現金'}")
print(f"\n  健康度(90日命中率):  A {hitA:.1f}% {light(hitA)}   趨勢 {hitT:.1f}% {light(hitT)}")
# Sharpe 誤差
if st['ndays']>5:
    hist=pd.read_csv(LOG,encoding='utf-8-sig')
    rets=pd.Series(hist['總權益$'].values).pct_change().dropna()
    if len(rets)>2 and rets.std()>0:
        SR=(rets.mean()*365)/(rets.std()*np.sqrt(365)); yrs=st['ndays']/365
        se=np.sqrt((1+0.5*SR**2)/yrs)
        print(f"\n  Forward Sharpe {SR:.2f} ± {se:.1f}  (才{yrs:.2f}年 → 誤差巨大, 現在的數字沒意義)")
print(f"  ★ 慢性衰退要~3年才看得出。別因短期好壞加碼或砍策略。")
print(f"\n  (state已存 paper_state.json — 過去紀錄不可竄改)")

# ================= 手機版儀表板 index.html =================
hist=pd.read_csv(LOG,encoding='utf-8-sig')
eqs=[float(x) for x in hist['總權益$']]
labs=[str(x) for x in hist['日K']]
peak=np.maximum.accumulate(eqs); dds=[round((e/p-1)*100,2) for e,p in zip(eqs,peak)]
ret_pct=(st['equity']/CAPITAL-1)*100
srtxt="樣本太少,還算不出"
if st['ndays']>5:
    rr=pd.Series(eqs).pct_change().dropna()
    if len(rr)>2 and rr.std()>0:
        SR=(rr.mean()*365)/(rr.std()*np.sqrt(365)); yy=st['ndays']/365
        srtxt=f"{SR:.2f} ± {np.sqrt((1+0.5*SR**2)/yy):.1f}(誤差巨大,別看)"
def lchip(h):
    c='#22c55e' if h>=52 else '#f59e0b' if h>=48 else '#ef4444'
    t='綠燈' if h>=52 else '黃燈' if h>=48 else '紅燈'
    return f'<span class="pill" style="background:{c}22;color:{c};border:1px solid {c}55">{h:.1f}% {t}</span>'
prows=""
for c in AP.columns:
    a=float(AP.loc[d,c])/len(AP.columns); t=float(TP.loc[d,c])/len(TP.columns) if c in TP.columns else 0.0
    net=(WA*a+(1-WA)*t)*100
    col='#22c55e' if net>0 else '#ef4444' if net<0 else '#8aa0b8'
    lab='淨多' if net>0 else '淨空' if net<0 else '空手'
    prows+=f'<div class="prow"><span>{c}</span><b style="color:{col}">{lab} {net:+.1f}%</b></div>'
J=json.dumps(dict(l=labs,e=eqs,d=dds),ensure_ascii=False)
LIVE=json.dumps(dict(refA=refA,refT=refT,wA=st['wA'],wT=st['wT'],
                      eqA=st['eqA'],eqT=st['eqT'],lockedEq=st['equity'],capital=CAPITAL),
                 ensure_ascii=False)

# ---- 新增①: 跟理論回測(combo_forward_log.csv)對照 ----
theory_html=""
CFL=os.path.join(HERE,'combo_forward_log.csv')
if os.path.exists(CFL):
    try:
        cf=pd.read_csv(CFL,encoding='utf-8-sig')
        if len(cf)>0:
            last_theory=cf.iloc[-1]
            t_val=float(last_theory['組合$'])
            t_date=str(last_theory['日K基準'])
            gap=st['equity']-t_val
            gap_pct=(gap/t_val*100) if t_val else 0
            gcol='#22c55e' if gap>=0 else '#ef4444'
            theory_html=f'''<div class="card"><div class="ct">跟理論回測對照(日K {t_date})</div>
 <div class="hrow"><span>真模擬金(含真實成本)</span><b>${st['equity']:,.0f}</b></div>
 <div class="hrow"><span>理論回測(簡化成本模型)</span><b>${t_val:,.0f}</b></div>
 <div class="hrow" style="border-top:1px solid #1e2a3a;margin-top:4px;padding-top:8px">
 <span>差額</span><b style="color:{gcol}">{gap:+,.0f} ({gap_pct:+.2f}%)</b></div></div>'''
    except Exception as e:
        theory_html=""

# ---- 新增②: 幣安測試網 實際帳戶(讀 bot_state.json 的快照, 可能落後一次執行) ----
bot_html=""
BOT_STATE=os.path.join(HERE,'bot_state.json')
if os.path.exists(BOT_STATE):
    try:
        bs=json.load(open(BOT_STATE,encoding='utf-8'))
        if 'equity' in bs:
            b_eq=bs['equity']; b_wallet=bs.get('wallet',b_eq); b_upnl=bs.get('upnl',0)
            b_time=bs.get('snapshot_utc','?')
            prows_bot=""
            for p in bs.get('positions',[]):
                pcol='#22c55e' if p['upnl']>=0 else '#ef4444'
                sym=p['symbol'].replace('USDT','')
                prows_bot+=f'<div class="prow"><span>{sym} {p["qty"]:+.4g}</span><b style="color:{pcol}">{p["upnl"]:+.2f}U ({p["pct"]:+.1f}%)</b></div>'
            if not prows_bot:
                prows_bot='<div class="prow"><span style="color:#5f7590">目前無持倉</span></div>'
            bot_html=f'''<div class="card"><div class="ct">幣安測試網 實際帳戶(快照 {b_time} UTC)</div>
 <div class="hrow"><span>權益</span><b>{b_eq:,.2f} USDT</b></div>
 <div class="hrow"><span>錢包 / 未實現</span><b>{b_wallet:,.2f} / {b_upnl:+,.2f}</b></div>
 {prows_bot}</div>'''
    except Exception as e:
        bot_html=""

# ---- 新增③: 最近逐筆交易紀錄(讀 orders_log.csv, 顯示最近10筆) ----
trades_html=""
ORDERS_LOG=os.path.join(HERE,'orders_log.csv')
if os.path.exists(ORDERS_LOG):
    try:
        ol=pd.read_csv(ORDERS_LOG,encoding='utf-8-sig')
        if len(ol)>0:
            recent=ol.tail(10).iloc[::-1]
            rows=""
            for _,r_ in recent.iterrows():
                side_col='#22c55e' if r_['方向']=='BUY' else '#ef4444'
                sym=str(r_['幣種']).replace('USDT','')
                rows+=(f'<div class="prow"><span>{r_["日K"]} {sym}</span>'
                       f'<b style="color:{side_col}">{r_["方向"]} {r_["數量"]}　${r_["成交價"]:,.4g}</b></div>')
            trades_html=f'<div class="card"><div class="ct">最近逐筆交易(近{len(recent)}筆)</div>{rows}</div>'
    except Exception as e:
        trades_html=""
html=f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>組合策略 模擬金</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#0b0f17;color:#e5edf7;
font-family:-apple-system,"Segoe UI","Microsoft JhengHei",sans-serif;padding:14px;max-width:560px;margin:0 auto}}
h1{{font-size:15px;margin:0 0 2px;color:#8aa0b8;font-weight:600}}
.date{{font-size:11px;color:#5f7590;margin-bottom:12px}}
.big{{background:#131a26;border-radius:14px;padding:16px;margin-bottom:12px;text-align:center}}
.eq{{font-size:34px;font-weight:700;letter-spacing:-1px}}
.rt{{font-size:15px;font-weight:600;margin-top:2px}}
.card{{background:#131a26;border-radius:14px;padding:14px;margin-bottom:12px}}
.ct{{font-size:12px;color:#8aa0b8;margin-bottom:8px;font-weight:600}}
.hrow{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;font-size:13px}}
.pill{{padding:3px 9px;border-radius:20px;font-size:11.5px;font-weight:600}}
.prow{{display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid #1e2a3a}}
.prow:last-child{{border:none}}
.wrap{{position:relative;height:170px}}
.note{{background:#2a2213;border-left:3px solid #f59e0b;border-radius:8px;padding:10px 12px;font-size:11.5px;color:#e8dcc0;line-height:1.6}}
.foot{{text-align:center;color:#5f7590;font-size:10.5px;margin-top:14px}}
.live{{margin-top:12px;padding-top:10px;border-top:1px dashed #2a3a4f}}
.live-lbl{{font-size:10.5px;color:#5f7590;margin-bottom:2px}}
.live-eq{{font-size:17px;font-weight:600;color:#60a5fa}}
.live-sub{{font-size:11px;color:#5f7590;margin-top:1px}}
</style></head><body>
<h1>組合策略 · 真模擬金</h1>
<div class="date">日K {d.date()} · 起算 {st['start']} · 第 {st['ndays']} 個交易日</div>
<div class="big"><div class="eq">${st['equity']:,.0f}</div>
 <div class="rt" style="color:{'#22c55e' if ret_pct>=0 else '#ef4444'}">{ret_pct:+.2f}%</div>
 <div class="live">
   <div class="live-lbl">⚡ 即時預覽(用現在市價估算,尚未收盤,不寫入帳本)</div>
   <div class="live-eq" id="liveEq">讀取中…</div>
   <div class="live-sub" id="liveDelta"></div>
 </div>
</div>
<div class="card"><div class="ct">權益曲線</div><div class="wrap"><canvas id="c"></canvas></div></div>
<div class="card"><div class="ct">健康度(90日命中率)</div>
 <div class="hrow"><span>A引擎(CB溢價)</span>{lchip(hitA)}</div>
 <div class="hrow"><span>B引擎(SMA50趨勢)</span>{lchip(hitT)}</div></div>
<div class="card"><div class="ct">今日淨部位</div>{prows}
 <div class="hrow" style="border-top:1px solid #1e2a3a;margin-top:6px;padding-top:8px">
 <span>A腿淨曝險</span><b>{sum(st['wA'].values())*100:+.0f}%</b></div>
 <div class="hrow"><span>趨勢腿投入</span><b>{sum(st['wT'].values())*100:.0f}%</b></div></div>
<div class="card"><div class="ct">Forward Sharpe</div>
 <div class="hrow"><span>估計值</span><b>{srtxt}</b></div></div>
{theory_html}
{bot_html}
{trades_html}
<div class="note"><b>別被短期數字騙:</b>慢性衰退要~3年才看得出,現在的Sharpe誤差巨大。紅綠燈連續兩月轉紅才是真警訊。押多少 &gt; 用什麼策略。</div>
<div class="foot">自動更新 · 過去紀錄不可竄改</div>
<script>
const D={J};
new Chart(document.getElementById('c'),{{type:'line',data:{{labels:D.l,datasets:[
{{data:D.e,borderColor:'#22c55e',borderWidth:2,pointRadius:0,fill:true,backgroundColor:'rgba(34,197,94,.08)',tension:.2}}]}},
options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},
tooltip:{{callbacks:{{label:c=>'$'+c.parsed.y.toFixed(0)}}}}}},
scales:{{y:{{grid:{{color:'#1e2a3a'}},ticks:{{color:'#5f7590',font:{{size:10}}}}}},
x:{{grid:{{display:false}},ticks:{{color:'#5f7590',font:{{size:9}},maxTicksLimit:5}}}}}}}}}});

// ---- 即時預覽: 用瀏覽器直接打幣安公開行情(不需金鑰), 對照昨天鎖定的部位現場估算 ----
const LIVE={LIVE};
async function updateLive(){{
  try{{
    const syms=[...new Set([...Object.keys(LIVE.refA),...Object.keys(LIVE.refT)])];
    const prices={{}};
    await Promise.all(syms.map(async s=>{{
      const r=await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${{s}}USDT`);
      const j=await r.json();
      prices[s]=parseFloat(j.price);
    }}));
    let rA=0; for(const c in LIVE.wA){{ if(LIVE.refA[c] && prices[c]) rA+=LIVE.wA[c]*(prices[c]/LIVE.refA[c]-1); }}
    let rT=0; for(const c in LIVE.wT){{ if(LIVE.refT[c] && prices[c]) rT+=LIVE.wT[c]*(prices[c]/LIVE.refT[c]-1); }}
    const liveEq=LIVE.eqA*(1+rA)+LIVE.eqT*(1+rT);
    const delta=liveEq-LIVE.lockedEq, dpct=delta/LIVE.lockedEq*100;
    document.getElementById('liveEq').textContent='$'+liveEq.toLocaleString(undefined,{{maximumFractionDigits:0}});
    const el=document.getElementById('liveDelta');
    el.textContent=(delta>=0?'+':'')+delta.toFixed(0)+' ('+(dpct>=0?'+':'')+dpct.toFixed(2)+'%) 較昨日鎖定值';
    el.style.color=delta>=0?'#22c55e':'#ef4444';
  }}catch(e){{
    document.getElementById('liveEq').textContent='無法取得即時價(可能離線)';
  }}
}}
updateLive();
setInterval(updateLive,8000);
</script></body></html>'''
open(os.path.join(HERE,'index.html'),'w',encoding='utf-8').write(html)
print(f"  📱 手機儀表板已生成 index.html")

# ================= Discord 推播 (只在有新K時) =================
WEBHOOK=os.environ.get('DISCORD_WEBHOOK','')
FORCE=os.environ.get('DISCORD_TEST','').lower()=='true'   # 手動測試推播用
if WEBHOOK and (todo or FORCE):
    def emo(h): return '🟢' if h>=52 else '🟡' if h>=48 else '🔴'
    lines=[]
    if FORCE and not todo: lines.append("🧪 *(測試推播 — 今日無新日K)*")
    lines+=[f"**日K {d.date()}** · 第{st['ndays']}日",
           f"💰 權益 **${st['equity']:,.2f}** ({ret_pct:+.2f}%)",
           f"{emo(hitA)} A引擎 {hitA:.1f}%　{emo(hitT)} 趨勢 {hitT:.1f}%",
           f"📊 A腿淨曝險 {sum(st['wA'].values())*100:+.0f}% · 趨勢投入 {sum(st['wT'].values())*100:.0f}%",
           f"📱 <https://pidasin.github.io/crypto_combo_forward/>"]
    if hitA<48 or hitT<48: lines.append("⚠️ **有引擎跌破48% — 若連續兩月則減碼**")
    try:
        requests.post(WEBHOOK,json={'content':"\n".join(lines)},timeout=15)
        print("  💬 已推播到 Discord")
    except Exception as e:
        print(f"  Discord 推播失敗: {e}")
