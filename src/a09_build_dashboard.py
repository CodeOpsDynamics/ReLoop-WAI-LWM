"""09 - Build the self-contained ReLoop Control Tower dashboard (HTML + Chart.js) from outputs/results.json."""
import json, pandas as pd
from config import RESULTS, TAB, ROOT

R = json.loads(RESULTS.read_text())
mm = pd.read_csv(TAB / "t01_mismatch_two_proxies.csv").sort_values("gen_share_internet", ascending=False)
mm = mm[(mm.gen_share_internet > 0.02) | (mm.proc_share > 0.02)]
ut = mm[mm.capacity_tpa_2026 > 5000].sort_values("utilisation_fy25", ascending=False)
lf = pd.read_csv(TAB / "t02_logistic_forecast.csv"); pf = pd.read_csv(TAB / "t02_policy_forecast.csv")
nat = pd.read_csv(ROOT / "data" / "raw" / "national_series.csv")
fm = pd.read_csv(TAB / "t06_fuzzy_fmea.csv")
data = dict(
    mm=dict(states=mm.label.tolist(), gen=(mm.gen_share_internet * 100).round(1).tolist(), proc=(mm.proc_share * 100).round(1).tolist(),
            mi=mm.MI_internet.round(2).tolist()),
    ut=dict(states=ut.label.tolist(), util=(ut.utilisation_fy25 * 100).round(1).tolist(), cap=(ut.capacity_tpa_2026 / 1e5).round(2).tolist()),
    fc=dict(years=lf.fy_end_year.tolist(), logistic=lf.logistic_lakh_t.round(2).tolist(),
            actual={int(a): round(b / 1e5, 2) for a, b in zip(nat.fy_end_year, nat.formally_processed_t)},
            base={int(r.fy_end_year): round(r.formal_flow_lakh_t, 2) for r in pf[pf.scenario.str.startswith("Base")].itertuples()},
            high={int(r.fy_end_year): round(r.formal_flow_lakh_t, 2) for r in pf[pf.scenario.str.startswith("High")].itertuples()},
            low={int(r.fy_end_year): round(r.formal_flow_lakh_t, 2) for r in pf[pf.scenario.str.startswith("Low")].itertuples()}),
    ml=R["disposition_ml"], net=R["network"], sens=R["network_sensitivity"], vrp=R["milkrun"],
    fmea=fm[["id", "failure_mode", "FRPN", "crisp_RPN", "S"]].to_dict("records"), cba=R["cba"], sm=R["spatial_mismatch"])

html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>ReLoop Control Tower | E-waste reverse logistics</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
<script>__CHARTJS__</script>
<style>
:root{--bg:#F4F7F5;--panel:#FFFFFF;--ink:#14261C;--muted:#5B6B62;--line:#DCE5DF;--dark:#1B4332;--green:#2D6A4F;--mid:#52B788;--light:#B7E4C7;--accent:#E76F51;--amber:#F4A261;
box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0F1A14;--panel:#16251C;--ink:#E3EEE7;--muted:#9DB3A6;--line:#26392E}}
:root[data-theme="dark"]{--bg:#0F1A14;--panel:#16251C;--ink:#E3EEE7;--muted:#9DB3A6;--line:#26392E}
*{box-sizing:border-box}html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
header{background:var(--dark);color:#EAF4EE;padding:28px clamp(16px,4vw,48px) 22px}
header h1{margin:0;font-size:clamp(22px,3vw,32px);font-weight:600;letter-spacing:-.01em}
header p{margin:6px 0 0;max-width:820px;color:#BFD8C9;font-size:15px}
.meta{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:#8FB9A2;margin-top:10px}
main{padding:20px clamp(16px,4vw,48px) 40px;max-width:1320px;margin:0 auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:-36px 0 22px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:22px;font-weight:600;color:var(--green);font-family:"IBM Plex Mono",ui-monospace,monospace}
.kpi span{font-size:13px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:16px}
section.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
section.card h2{font-size:16px;margin:0 0 2px;font-weight:600}
section.card .so{font-size:13px;color:var(--muted);margin:0 0 10px}
.cw{position:relative;height:300px}
.whatif{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:center}
.whatif label{font-size:13px;color:var(--muted);display:block;margin-top:10px}
input[type=range]{width:100%;accent-color:var(--green)}
.out{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:22px;color:var(--dark)}
:root[data-theme="dark"] .out{color:var(--mid)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--muted);font-weight:500}
.tw{overflow-x:auto}
footer{font-size:12px;color:var(--muted);margin-top:24px;max-width:900px}
.toggle{float:right;background:transparent;border:1px solid #4E7A62;color:#CFE6D8;border-radius:6px;padding:4px 10px;font:inherit;font-size:12px;cursor:pointer}
@media (max-width:640px){.whatif{grid-template-columns:1fr}.cw{height:260px}}
</style></head><body>
<header><button class="toggle" id="tg">Toggle theme</button>
<h1>ReLoop Control Tower</h1>
<p>AI-enabled reverse logistics for corporate IT assets and e-waste in India: where the flow goes today, what the network should look like, and what it is worth.</p>
<div class="meta">WAI project | Logistics &amp; Warehousing Management | IIM Ranchi EMBA 2025-27 | Himanshu Rai</div>
</header>
<main>
<div class="kpis" id="kpis"></div>
<div class="grid">
<section class="card"><h2>1. The mismatch: generated in the south and west, processed in the north</h2><p class="so">Share of national total, FY 2024-25. Generation proxy: internet subscribers (TRAI x Census); processing: CPCB actuals.</p><div class="cw"><canvas id="c1"></canvas></div></section>
<section class="card"><h2>1b. Capacity is not the bottleneck: India used 23% of it in FY25</h2><p class="so">FY25 processed / authorised capacity (CPCB registry, 411 recyclers, Aug 2026). Hover for capacity.</p><div class="cw"><canvas id="c1b"></canvas></div></section>
<section class="card"><h2>2. Formal flow will roughly double by FY30</h2><p class="so">Lakh tonnes per year. Data-only S-curve vs. EPR-target-driven scenarios.</p><div class="cw"><canvas id="c2"></canvas></div></section>
<section class="card"><h2>3. AI grading recovers more value per returned asset</h2><p class="so">Average INR recovered per asset on a 3,000-asset hold-out set.</p><div class="cw"><canvas id="c3"></canvas></div></section>
<section class="card"><h2>4. Three regional hubs beat one national vendor</h2><p class="so">Annual cost (INR crore, bars) and CO2e (tonnes, line) for the 2,400 t/yr portfolio.</p><div class="cw"><canvas id="c4"></canvas></div></section>
<section class="card"><h2>5. Risk priorities from fuzzy FMEA</h2><p class="so">Defuzzified fuzzy RPN vs. conventional crisp RPN.</p><div class="cw"><canvas id="c5"></canvas></div></section>
<section class="card"><h2>6. Milk-runs cut first-mile kilometres in Pune</h2><p class="so">Vehicle-km per week, 20 client sites, secure 1.8 t vans.</p><div class="cw"><canvas id="c6"></canvas></div></section>
<section class="card"><h2>7. What-if: how much of the AI value uplift do we need to believe?</h2><p class="so">5-year NPV at 12% recomputed live from the model outputs.</p>
<div class="whatif"><div>
<label for="s1">Share of modelled value uplift realised: <b id="v1"></b></label><input id="s1" type="range" min="0" max="100" step="5" value="50">
<label for="s2">Freight savings realised: <b id="v2"></b></label><input id="s2" type="range" min="50" max="130" step="5" value="100">
<label for="s3">Capex overrun: <b id="v3"></b></label><input id="s3" type="range" min="0" max="60" step="5" value="0">
</div><div><div class="so">5-year NPV</div><div class="out" id="npv"></div><div class="so" style="margin-top:10px">Payback</div><div class="out" id="pb"></div></div></div></section>
<section class="card"><h2>8. Robustness of the hub decision</h2><p class="so">MILP re-solved under cost shocks.</p><div class="tw"><table id="t8"></table></div></section>
</div>
<footer>Sources: CPCB recycler registry (Aug 2026); PIB releases 1943201, 2147876; Lok Sabha USQ 983 (2020) and 2458 (2021); TRAI via Jharkhand Economic Survey 2025-26; Census 2011. Asset-grading records and client volumes are synthetic and disclosed; FMEA panel simulated; cost parameters are stated assumptions. Code: run_all.py in the project repository reproduces every number.</footer>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const charts=[];
function base(){Chart.defaults.font.family='"IBM Plex Sans", system-ui, sans-serif';Chart.defaults.color=css('--muted');Chart.defaults.borderColor=css('--line');}
function kpis(){
 const k=[[D.sm.north_belt_share_pct+'%','of formal processing in UP, Uttarakhand and Haryana'],
 [D.sm.national_utilisation_pct+'%','of authorised recycling capacity used in FY25 (411 recyclers)'],
 [D.net.S2_vs_S0_cost_saving_pct+'%','lower annual logistics cost (optimised vs. status quo)'],
 [D.net.S2_vs_S0_co2_saving_pct+'%','lower transport CO2e'],
 [D.net.S2_vs_S0_data_exposure_cut_pct+'%','less data-bearing tonne-km in transit'],
 ['+Rs '+D.ml.uplift_per_asset.toLocaleString('en-IN'),'value recovered per asset with AI grading'],
 ['Rs '+D.cba.base_npv_cr.toFixed(1)+' Cr','5-year NPV, base case']];
 document.getElementById('kpis').innerHTML=k.map(x=>`<div class="kpi"><b>${x[0]}</b><span>${x[1]}</span></div>`).join('');}
function draw(){
 charts.forEach(c=>c.destroy());charts.length=0;base();
 const G=css('--green'),A=css('--amber'),AC=css('--accent'),M=css('--mid'),L=css('--light'),DK=css('--dark');
 charts.push(new Chart(c1,{type:'bar',data:{labels:D.mm.states,datasets:[{label:'Est. generation share %',data:D.mm.gen,backgroundColor:A},{label:'Formal processing share %',data:D.mm.proc,backgroundColor:G}]},options:{indexAxis:'y',maintainAspectRatio:false,scales:{y:{ticks:{autoSkip:false,font:{size:11}}}},plugins:{tooltip:{callbacks:{afterBody:(i)=>'Mismatch index: '+D.mm.mi[i[0].dataIndex]}}}}}));
 charts.push(new Chart(c1b,{type:'bar',data:{labels:D.ut.states,datasets:[{label:'Utilisation %',data:D.ut.util,backgroundColor:D.ut.util.map(v=>v<15?AC:G)}]},options:{indexAxis:'y',maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{afterBody:(i)=>'Capacity: '+D.ut.cap[i[0].dataIndex]+' lakh t'}}},scales:{x:{max:100,title:{display:true,text:'% of authorised capacity used'}},y:{ticks:{autoSkip:false}}}}}));
 const yrs=D.fc.years, pick=o=>yrs.map(y=>o[y]??null);
 charts.push(new Chart(c2,{type:'line',data:{labels:yrs,datasets:[{label:'Actual (CPCB)',data:pick(D.fc.actual),borderColor:DK,backgroundColor:DK,showLine:false,pointRadius:5},{label:'Logistic S-curve',data:D.fc.logistic,borderColor:G,pointRadius:0,tension:.3},{label:'Policy: low',data:pick(D.fc.low),borderColor:AC,borderDash:[2,3],pointRadius:0},{label:'Policy: base',data:pick(D.fc.base),borderColor:AC,borderDash:[6,4],pointRadius:0},{label:'Policy: high',data:pick(D.fc.high),borderColor:AC,pointRadius:0}]},options:{maintainAspectRatio:false,spanGaps:true}}));
 charts.push(new Chart(c3,{type:'bar',data:{labels:['Rule: age>=4 to recycler','AI grading (Random Forest)','Perfect information'],datasets:[{data:[D.ml.value_rule,D.ml.value_ml,D.ml.value_perfect],backgroundColor:[A,G,L]}]},options:{maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{title:{display:true,text:'INR per asset'}}}}}));
 const n=D.net;charts.push(new Chart(c4,{data:{labels:['S0 Central vendor','S1 Hubs + north recyclers','S2 Optimised network'],datasets:[{type:'bar',label:'Cost (INR Cr)',data:[n.S0.cost_cr,n.S1.cost_cr,n.S2.cost_cr],backgroundColor:[A,M,G],yAxisID:'y'},{type:'line',label:'CO2e (t)',data:[n.S0.co2_t,n.S1.co2_t,n.S2.co2_t],borderColor:DK,backgroundColor:DK,yAxisID:'y2'}]},options:{maintainAspectRatio:false,scales:{y:{beginAtZero:true,title:{display:true,text:'INR crore'}},y2:{position:'right',beginAtZero:true,grid:{display:false},title:{display:true,text:'t CO2e'}}}}}));
 charts.push(new Chart(c5,{type:'bar',data:{labels:D.fmea.map(r=>r.id+' '+r.failure_mode.slice(0,34)),datasets:[{label:'Fuzzy RPN',data:D.fmea.map(r=>r.FRPN),backgroundColor:G},{label:'Crisp RPN',data:D.fmea.map(r=>r.crisp_RPN),backgroundColor:A}]},options:{indexAxis:'y',maintainAspectRatio:false,scales:{y:{ticks:{autoSkip:false,font:{size:10}}}}}}));
 charts.push(new Chart(c6,{type:'bar',data:{labels:['Ad-hoc trip per ticket','Optimised milk-run'],datasets:[{data:[D.vrp.baseline_km,D.vrp.optimised_km],backgroundColor:[A,G]}]},options:{maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{title:{display:true,text:'vehicle-km / week'}}}}}));
}
function whatif(){
 const vs=s1.value/100, fs=s2.value/100, co=1+s3.value/100;
 v1.textContent=s1.value+'%';v2.textContent=s2.value+'%';v3.textContent='+'+s3.value+'%';
 const capex=D.cba.capex_total*co, opex=Object.values(D.cba.opex_add).reduce((a,b)=>a+b,0);
 const ramp=[.4,.7,.85,.85,.85]; let npv=-capex, cum=-capex, pb=null;
 ramp.forEach((r,i)=>{const net=r*(fs*(D.cba.net_network_saving_cr+D.cba.milkrun_saving_cr)+vs*D.cba.value_uplift_full_cr)-opex;
  npv+=net/Math.pow(1.12,i+1); const prev=cum; cum+=net; if(pb===null&&prev<0&&cum>=0) pb=12*(i+(-prev)/net);});
 document.getElementById('npv').textContent='Rs '+npv.toFixed(1)+' crore';
 document.getElementById('pb').textContent=pb===null?'beyond 5 years':pb.toFixed(1)+' months';}
function table(){t8.innerHTML='<tr><th>Case</th><th>Hubs opened</th><th>Status quo (Cr)</th><th>Optimised (Cr)</th><th>Saving</th></tr>'+D.sens.map(r=>`<tr><td>${r.case}</td><td>${r.hubs}</td><td>${r.s0_cr}</td><td>${r.s2_cr}</td><td>${r.saving_pct}%</td></tr>`).join('');}
[s1,s2,s3].forEach(e=>e.addEventListener('input',whatif));
tg.onclick=()=>{const r=document.documentElement;const cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');r.setAttribute('data-theme',cur==='dark'?'light':'dark');draw();};
kpis();draw();whatif();table();
</script></body></html>"""
out = ROOT / "dashboard" / "ReLoop_Control_Tower.html"
chartjs = (ROOT / "dashboard" / "vendor_chart.umd.min.js").read_text()   # Chart.js 4.4.1 (MIT) inlined -> works offline
out.write_text(html.replace("__DATA__", json.dumps(data, default=float)).replace("__CHARTJS__", chartjs.replace("</script", "<\\/script")))
print("dashboard written", out)
