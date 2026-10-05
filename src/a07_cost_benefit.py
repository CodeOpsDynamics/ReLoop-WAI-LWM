"""
07 - Cost-benefit analysis, NPV and scenario stress test of the ReLoop programme (5-year horizon, 12% WACC).
Benefits are pulled from modules 03-05 (single source of truth: outputs/results.json) and haircut conservatively.
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from config import RESULTS, FIG, TAB, C, save_result

R = json.loads(RESULTS.read_text())
WACC, YEARS = 0.12, 5
capex = {"3 regional ITAD hubs (secure cage, racking, wipe stations, shredder)": 3.30,
         "AI grading + asset-tracking platform (QR/RFID, control tower)": 0.90,
         "GPS e-locks and IoT trackers for secure vans": 0.25,
         "Training, SOPs, R2/ISO 27001 certification readiness": 0.20}
capex["Contingency (10%)"] = round(0.10 * sum(capex.values()), 2)
opex_add = {"Platform run cost + cloud": 0.25, "Data science / analytics (2 FTE)": 0.50}
ramp = [0.40, 0.70, 0.85, 0.85, 0.85]                  # realisation of steady-state benefits

net_saving = R["network"]["S2_vs_S0_cost_saving_cr"]
milk_per_t = R["milkrun"]["annual_cost_saving_lakh"] * 1e5 / (R["milkrun"]["total_kg"] * 50 / 1000)
milk = milk_per_t * R["network"]["total_tonnes"] / 1e7
value_uplift_full = R["network"]["value_uplift_cr"]
HAIRCUT = 0.50                                         # synthetic-data model -> count only half of modelled uplift

def run(value_share, freight_share=1.0):
    rows, cum = [], -sum(capex.values())
    rows.append(dict(year=0, capex=-sum(capex.values()), benefit=0, opex=0, net=-sum(capex.values()), cum=cum))
    for y in range(1, YEARS + 1):
        b = ramp[y - 1] * (freight_share * (net_saving + milk) + value_share * value_uplift_full)
        o = sum(opex_add.values())
        net = b - o; cum += net
        rows.append(dict(year=y, capex=0, benefit=b, opex=-o, net=net, cum=cum))
    df = pd.DataFrame(rows)
    npv = sum(r.net / (1 + WACC) ** r.year for r in df.itertuples())
    # payback (months) via linear interpolation of cumulative cash
    pb = None
    for i in range(1, len(df)):
        if df.cum[i - 1] < 0 <= df.cum[i]:
            pb = 12 * (i - 1 + (-df.cum[i - 1]) / df.net[i]); break
    return df, npv, pb

base_df, base_npv, base_pb = run(HAIRCUT)
base_df.to_csv(TAB / "t07_cashflow_base.csv", index=False)
scen = []
for name, vs, fs in [("Logistics only\n(no value uplift)", 0.0, 1.0), ("Conservative\n(25% uplift, 80% freight)", 0.25, 0.8),
                     ("Base\n(50% uplift)", 0.50, 1.0), ("Full modelled\nuplift", 1.0, 1.0)]:
    _, npv, pb = run(vs, fs)
    scen.append(dict(scenario=name.replace("\n", " "), label=name, npv_cr=round(npv, 2), payback_months=None if pb is None else round(pb, 1)))
pd.DataFrame(scen).drop(columns="label").to_csv(TAB / "t07_scenarios.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
ax.bar(base_df.year, base_df.net, color=[C["accent"] if v < 0 else C["green"] for v in base_df.net])
ax.plot(base_df.year, base_df.cum, "-o", color=C["dark"], label="Cumulative cash")
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Year"); ax.set_ylabel("INR crore")
ax.set_title(f"Fig 7a. Base case cash flow (NPV Rs {base_npv:.1f} Cr)"); ax.legend(frameon=False)
ax = axes[1]
sd = pd.DataFrame(scen)
ax.barh(sd.label, sd.npv_cr, height=0.6, color=[C["amber"], C["mid"], C["green"], C["dark"]])
for i, r in sd.iterrows():
    ax.text(r.npv_cr + 0.5, i, f"Rs {r.npv_cr:.1f} Cr | payback {r.payback_months if r.payback_months else '>60'} m", va="center", fontsize=8)
ax.tick_params(axis="y", labelsize=8); ax.set_xlabel("5-year NPV @12% (INR crore)"); ax.set_title("Fig 7b. NPV under benefit scenarios")
ax.set_xlim(min(0, sd.npv_cr.min()) - 1, sd.npv_cr.max() * 1.6)
fig.savefig(FIG / "fig07_cost_benefit.png"); plt.close(fig)

scen_out = [{k: v for k, v in d.items() if k != "label"} for d in scen]
res = dict(capex=capex, capex_total=round(sum(capex.values()), 2), opex_add=opex_add,
           net_network_saving_cr=net_saving, milkrun_saving_cr=round(milk, 2), value_uplift_full_cr=value_uplift_full,
           base_npv_cr=round(base_npv, 2), base_payback_months=round(base_pb, 1), scenarios=scen_out,
           steady_state_benefit_base_cr=round(0.85 * (net_saving + milk + HAIRCUT * value_uplift_full), 2))
save_result("cba", res); print(json.dumps(res, indent=1))
