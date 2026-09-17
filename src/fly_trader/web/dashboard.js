const $=id=>document.getElementById(id);let paused=false,selected='',market='',manualMarket=false;
const names={BUY:'买入',SELL:'卖出',HOLD:'观望'};
function fmt(v,n=3){return Number.isFinite(v)?Number(v).toFixed(n):'—'}
function clock(zone){const p=Object.fromEntries(new Intl.DateTimeFormat('en-US',{timeZone:zone,weekday:'short',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts().filter(x=>x.type!=='literal').map(x=>[x.type,x.value]));return{day:p.weekday,minute:+p.hour*60+(+p.minute)}}
function open(c,ranges){return!['Sat','Sun'].includes(c.day)&&ranges.some(([a,b])=>c.minute>=a&&c.minute<b)}
function defaultMarket(){const hk=clock('Asia/Hong_Kong'),us=clock('America/New_York');if(open(hk,[[570,720],[780,960]]))return'HK';if(open(us,[[570,960]]))return'US';return hk.minute>=480&&hk.minute<1020?'HK':'US'}
function setMarket(value,manual=false){market=value;manualMarket=manual;selected='';document.querySelectorAll('.market-tab').forEach(b=>b.classList.toggle('active',b.dataset.market===market));$('marketHint').textContent=manual?'手动选择':'已按交易时段自动选择'}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function money(v,c){return Number.isFinite(v)?`${c||''} ${Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`:'—'}
function pct(v){return Number.isFinite(v)?`${v>=0?'+':''}${(v*100).toFixed(2)}%`:'—'}
function draw(series){const c=$('scoreChart'),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);x.strokeStyle='#243b35';[.2,.5,.8].forEach(p=>{x.beginPath();x.moveTo(0,h*p);x.lineTo(w,h*p);x.stroke()});if(series.length<2)return;const values=series.flatMap(p=>[p.raw_score||0,p.smooth_score||0]),max=Math.max(25,...values.map(Math.abs));const plot=(key,color)=>{x.strokeStyle=color;x.lineWidth=2;x.beginPath();series.forEach((p,i)=>{const px=i/(series.length-1)*w,py=h/2-(p[key]||0)/max*h*.43;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()};plot('raw_score','#486a60');plot('smooth_score','#66e3c4')}
let currentState=null, reportRun=null, refreshing=false, displayedRun=null;
const trackLabels={fly_raw:'果蝇原始',fly_filtered:'人工过滤',risk_executable:'风控可执行',buy_and_hold:'买入持有',disconnected_connectome:'断开连接组'};
const states={idle:'待配置',preparing:'准备中',running:'运行中',paused:'已暂停',stopping:'结束中',completed:'已完成',failed:'失败'};
const reasons={duration_elapsed:'时长已到',user_stopped:'手动结束',process_stopped:'程序停止',market_failed:'行情服务停止',preparation_failed:'准备失败'};
function portfolioTable(tracks){
  return Object.values(tracks||{}).map(a=>`<tr data-track="${esc(a.track)}" class="${a.track===$('trackSelector').value?'selected-track':''}"><th scope="row">${esc(a.label)}</th><td>${money(a.initial_cash)}</td><td>${money(a.equity)}</td><td class="${a.total_pl>=0?'positive':'negative'}">${money(a.total_pl)}</td><td class="${a.return>=0?'positive':'negative'}">${pct(a.return)}</td><td>${pct(a.max_drawdown)}</td><td>${fmt(a.bought_quantity,0)}</td><td>${fmt(a.sold_quantity,0)}</td><td>${a.trade_count}</td></tr>`).join('');
}
function recordTable(records,currency){
  return `<div class="table-scroll"><table class="holdings-table"><thead><tr><th>股票</th><th>累计买入 / 股</th><th>累计卖出 / 股</th><th>持有 / 股</th><th>持仓成本价</th><th>平均买入价</th><th>平均卖出价</th><th>持仓市值</th><th>已实现盈亏</th><th>未实现盈亏</th><th>净收益</th></tr></thead><tbody>${records.map(p=>`<tr><th scope="row" title="${esc(p.execution_note)}">${esc(p.symbol)}<small>${esc(p.current_price==null&&p.execution_note!=='现金对照，不交易'?'未收到有效行情，缺少成交价格':p.execution_note||'')}</small></th><td>${fmt(p.bought_quantity,0)}</td><td>${fmt(p.sold_quantity,0)}</td><td>${fmt(p.quantity,0)}</td><td>${fmt(p.cost_price,4)}</td><td>${fmt(p.average_buy_price,4)}</td><td>${fmt(p.average_sell_price,4)}</td><td>${money(p.market_value)}</td><td>${money(p.realized_pl)}</td><td>${money(p.unrealized_pl)}</td><td class="${p.total_pl>=0?'positive':'negative'}">${money(p.total_pl)}</td></tr>`).join('')||'<tr><td colspan="11">配置股票并开始实验后显示交易统计</td></tr>'}</tbody></table></div>`;
}
function renderAccount(tracks){
  const a=(tracks||{})[$('trackSelector').value];if(!a)return;
  $('accountProvider').textContent=`${a.label} · ${a.currency} · 独立初始资金 ${money(a.initial_cash)}`;
  for(const [id,key] of Object.entries({accountEquity:'equity',accountCash:'cash',accountBuyingPower:'buying_power',realizedPl:'realized_pl',unrealizedPl:'unrealized_pl'}))$(id).textContent=money(a[key],a.currency);
  $('positionCount').textContent=(a.positions||[]).length;
  $('positions').innerHTML=recordTable(a.symbol_records||[],a.currency);
}
function renderProposal(p){
  $('proposalAction').textContent=names[p?.action]||'观望';
  $('proposalQuantity').textContent=fmt(p?.quantity||0,0);
  $('proposalPrice').textContent=fmt(p?.reference_price);
  $('proposalReason').textContent=p?.reason||'等待行情与稳定信号';
  $('proposalStatus').textContent=({ready:'模拟成交通过',blocked:'已拦截',confirming:'确认中',observing:'观察中'})[p?.status]||'等待稳定信号';
  const icons={passed:'✓',blocked:'×',confirming:'…',observing:'…',skipped:'–',pending:'·'};
  $('riskChecks').innerHTML=(p?.risk_checks||[]).map(c=>`<span class="risk-${esc(c.status)}" title="${esc(c.reason)}">${icons[c.status]||'·'} ${esc(c.label)}</span>`).join('')||'等待风控检查';
}
function renderPerformance(tracks){
  const values=Object.values(tracks||{});
  $('performanceMeta').textContent=`${values[0]?.currency||market} · 每条轨道独立初始资金 1,000,000`;
  $('performanceRows').innerHTML=portfolioTable(tracks);
  $('performanceRows').querySelectorAll('tr').forEach(row=>row.onclick=()=>{$('trackSelector').value=row.dataset.track;renderState(currentState)});
}
function renderState(s){
  currentState=s;const e=s.experiment||{},busy=['preparing','running','paused','stopping'].includes(e.status);
  if(e.run_id&&displayedRun!==e.run_id){
    displayedRun=e.run_id;
    if(!(e.config.hk_symbols||[]).length)setMarket('US');
    else if(!(e.config.us_symbols||[]).length)setMarket('HK');
  }
  $('connection').textContent=states[e.status]||'连接中';
  $('experimentFields').disabled=busy;$('pause').disabled=!['running','paused'].includes(e.status);
  $('stop').disabled=!['preparing','running','paused'].includes(e.status);
  paused=e.status==='paused';$('pause').textContent=paused?'继续实验':'暂停实验';
  $('start').textContent=['completed','failed'].includes(e.status)?'开始新实验':'开始实验';
  $('countdown').textContent=['running','paused'].includes(e.status)?(e.remaining_s===null?'不限时':`剩余 ${Math.ceil(e.remaining_s)} 秒`):states[e.status]||'等待开始';
  $('experimentMessage').textContent=[...Object.values(e.errors||{}),...(e.warnings||[])].join('；');
  if(busy){$('usSymbols').value=(e.config.us_symbols||[]).join(', ');$('hkSymbols').value=(e.config.hk_symbols||[]).join(', ')}
  $('resultCard').hidden=!e.report_available;
  if(e.report_available&&reportRun!==e.run_id)loadReport(e.run_id);
  if(!e.report_available)reportRun=null;
  const all=s.assets||{},assets=Object.fromEntries(Object.entries(all).filter(([symbol,a])=>(a.market||(symbol.endsWith('.HK')?'HK':'US'))===market));
  if(!assets[selected])selected=Object.keys(assets)[0]||'';
  const v=assets[selected]||{},p=(s.proposals||{})[selected];
  for(const [id,action] of [['rawTrack',v.raw_action],['signal',v.stable_action],['executableTrack',p?.status==='ready'?p.side:'HOLD']]){
    $(id).textContent=names[action]||'观望';$(id).className=(action||'HOLD').toLowerCase();
  }
  $('source').textContent=selected||'—';$('feedName').textContent=v.source_name?.startsWith('longbridge-')?`长桥 · ${{pre:'盘前',regular:'交易时段',post:'盘后',overnight:'夜盘'}[v.trade_session]||'行情'}`:(v.source_name||'等待行情');$('price').textContent=fmt(v.price,4);
  $('age').textContent=Number.isFinite(v.market_time_ms)?`${fmt(Math.max(0,(Date.now()-v.market_time_ms)/1000),1)} 秒`:'—';
  $('rawAction').textContent=names[v.raw_action]||'—';
  for(const [id,key] of Object.entries({rawScore:'raw_score',smoothScore:'smooth_score',windowMean:'window_mean_score',enterThreshold:'enter_threshold',triggerDistance:'distance_to_trigger'}))$(id).textContent=fmt(v[key]);
  const hasQuote=Number.isFinite(v.price),samples=v.baseline_samples||0,minSamples=v.baseline_min_samples||(market==='HK'?30:100);
  const noData=!hasQuote&&!!selected;
  const timeLeft=Number.isFinite(v.warmup_remaining_s)&&!noData?v.warmup_remaining_s:Math.max(0,300-(v.elapsed_s||0));
  if(noData)$('calibration').textContent=`等待行情 · 有效样本 ${samples}/${minSamples}`;
  else if(v.calibration_status==='ready')$('calibration').textContent=`已完成 · ${samples} 样本${v.market_stale?' · 行情已过期':''}`;
  else if(timeLeft>0)$('calibration').textContent=`预热剩余 ${Math.ceil(timeLeft)} 秒 · 样本 ${samples}/${minSamples}`;
  else $('calibration').textContent=`时间已满足 · 等待样本 ${samples}/${minSamples}`;
  const noDataReason=(v.market_data_reason?[v.market_data_reason,...(v.source_status||[]).map(x=>x.reason)].filter(Boolean).join('；'):null)||'尚未收到有效行情，缺少成交价格；请检查数据源时段、连接和订阅权限。';
  $('marketNotice').hidden=!noData&&v.market_data_status!=='stale';
  $('marketNotice').textContent=noData?`${selected} 未收到行情，暂不能模拟买卖。${noDataReason} 原始信号仍可能来自神经背景活动，不表示已成交。`:(v.market_data_reason||'');
  $('rawSignalNote').textContent=noData?'无行情 · 仅神经输出':'';

  $('stimulus').textContent=Number.isFinite(v.stimulus_scale)?`${fmt(v.stimulus_scale*100,0)}%`:'—';$('steps').textContent=(v.model_step||0).toLocaleString();
  const h=v.runtime_health||{};$('eventHealth').textContent=`${h.received_events||0} / ${h.accepted_events||0}`;$('dropHealth').textContent=`${h.dropped_queue_events||0} / ${h.duplicate_or_old_events||0}`;$('latencyHealth').textContent=fmt(h.max_tick_latency_ms,1)+' ms';
  $('assets').innerHTML=Object.values(assets).map(a=>`<button class="asset ${a.symbol===selected?'active':''}" data-symbol="${esc(a.symbol)}"><b>${esc(a.symbol)}</b><em class="${(a.stable_action||'HOLD').toLowerCase()}">${names[a.stable_action]||'观望'}</em><span>${fmt(a.price,2)}</span></button>`).join('');
  document.querySelectorAll('.asset').forEach(b=>b.onclick=()=>{selected=b.dataset.symbol;renderState(currentState)});
  renderAccount((s.track_accounts||{})[market]);renderProposal(p);renderPerformance((s.track_accounts||{})[market]);draw((s.series||[]).filter(p=>p.symbol===selected));
}
async function loadReport(runId){
  reportRun=runId;
  try{
    const response=await fetch('/api/experiment/report');if(!response.ok)throw new Error('报告暂不可用');const r=await response.json();
    if(currentState?.experiment.run_id!==r.run_id)return;
    $('downloadReport').download=`${r.run_id}-report.json`;
    $('resultMeta').textContent=`${r.run_id} · ${reasons[r.reason]||r.reason} · ${r.complete?'已生成结果':'结果不完整'} · ${r.started_at?new Date(r.started_at).toLocaleString():'未开始'} → ${new Date(r.ended_at).toLocaleString()} · ${fmt(r.elapsed_s,1)} 秒`;
    $('resultAccounts').innerHTML=Object.entries(r.track_accounts||{}).map(([m,tracks])=>`<article class="result-account"><h3>${m==='US'?'美元':'港元'} · 各轨道结算</h3><div class="table-scroll"><table class="portfolio-table"><thead><tr><th>实验轨道</th><th>初始资金</th><th>总资产</th><th>净收益</th><th>收益率</th><th>最大回撤</th><th>累计买入 / 股</th><th>累计卖出 / 股</th><th>成交次数</th></tr></thead><tbody>${portfolioTable(tracks)}</tbody></table></div></article>`).join('');
    $('resultDiagnostics').innerHTML=Object.entries(r.symbols).map(([symbol,d])=>`<div>${esc(symbol)}：${!d.has_data?'无行情，未形成有效实验结果':`${d.warmup_complete?'预热完成':'预热未完成'} · 最后价格 ${fmt(d.last_price,4)} · ${esc(new Date(d.valuation_time_ms).toLocaleString())}${d.stale?' · 估值行情已过期':''}`}</div>`).join('');
    const trades=Object.values(r.track_trades||{}).flatMap(tracks=>Object.values(tracks).flat());
    $('resultTrades').innerHTML=trades.length?`<table><thead><tr><th>时间</th><th>轨道</th><th>股票</th><th>方向</th><th>数量</th><th>参考价</th><th>费用</th></tr></thead><tbody>${trades.map(t=>`<tr><td>${esc(new Date(t.timestamp).toLocaleString())}</td><td>${esc(trackLabels[t.track]||t.track)}</td><td>${esc(t.symbol)}</td><td>${names[t.side]}</td><td>${t.quantity}</td><td>${money(t.reference_price,t.currency)}</td><td>${money(t.costs,t.currency)}</td></tr>`).join('')}</tbody></table>`:'没有模拟成交';
  }catch(error){reportRun=null;$('formError').textContent=error.message}
}
async function request(path,body){
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const result=await response.json();if(!response.ok)throw new Error(result.error||'操作失败');return result;
}
async function action(path,body){
  $('formError').textContent='';try{await request(path,body);await refresh()}catch(error){$('formError').textContent=error.message}
}
async function refresh(){if(refreshing)return;refreshing=true;try{const r=await fetch('/api/state');if(!r.ok)throw new Error();renderState(await r.json())}catch(error){$('connection').textContent='连接断开'}finally{refreshing=false}}
$('experimentForm').onsubmit=event=>{event.preventDefault();action('/api/experiment/start',{us_symbols:$('usSymbols').value,hk_symbols:$('hkSymbols').value,duration_s:Number($('duration').value)*Number($('durationUnit').value)})};
$('pause').onclick=()=>action('/api/pause',{paused:!paused});$('stop').onclick=()=>action('/api/experiment/stop',{});
function durationHint(){$('durationHint').textContent=Number($('duration').value)*Number($('durationUnit').value)<300?'时长不足 5 分钟，人工过滤轨道可能无法完成预热。':''}
$('duration').oninput=durationHint;$('durationUnit').onchange=durationHint;
document.querySelectorAll('.market-tab').forEach(b=>b.onclick=()=>{setMarket(b.dataset.market,true);if(currentState)renderState(currentState)});
$('trackSelector').onchange=()=>{if(currentState)renderState(currentState)};
setMarket(defaultMarket());setInterval(refresh,500);refresh();
