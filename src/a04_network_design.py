"""
04 - Reverse-logistics network design for a corporate ITAD portfolio (BFSI / GCC clients, 12 Indian cities).
Step A (unsupervised ML): K-means on ~420 synthetic client pickup sites, weighted by tonnage, to discover
        natural consolidation regions -> validates the candidate hub list.
Step B (optimisation): Mixed-integer linear program (capacitated facility location, single sourcing) solved
        with Google OR-Tools (SCIP). Three scenarios isolate the value of each design lever:
          S0 Status quo  : every city ships retired assets LTL to ONE national ITAD vendor facility (Gautam Buddha Nagar,
                           wipe + grade + recycle on site) - the prevailing "single national vendor" model
          S1 Hub & spoke : sites consolidate to regional hubs (grade + wipe), FTL onward, NORTH recyclers only
          S2 Optimised   : hubs + regionally assigned recyclers (full MILP)
Concept link: Session 2 (3PL/4PL), Session 9 (freight cost, consolidation, mode), Session 12-13 (warehouse
site/type decisions), Session 7-8 (CO2), Plaza Zaragoza case (logistics-park consolidation logic).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ortools.linear_solver import pywraplp
from config import SYN, FIG, TAB, C, SEED, CITY, road_km, save_result
import json
from config import RESULTS

rng = np.random.default_rng(SEED)
ml = json.loads(RESULTS.read_text())["disposition_ml"]
ONWARD = ml["weight_share_recycle_or_harvest"]          # share of weight that must go to a recycler after grading

DEMAND = {"Bengaluru": 480, "Hyderabad": 330, "Pune": 260, "Mumbai": 300, "Chennai": 260, "Gurugram": 300,
          "Noida": 200, "Kolkata": 110, "Ahmedabad": 70, "Kochi": 40, "Jaipur": 30, "Bhubaneswar": 20}  # t/yr
# Fixed cost = rent (~15k sq ft), CCTV/secure cage, ISO 27001 / R2 certification, core supervisors (INR/yr); capacity t/yr.
# Volume-driven processing cost (wiping, grading labour) is identical per tonne in every scenario, so it cancels out.
HUBS = {"Bengaluru": (0.95e7, 1200), "Hyderabad": (0.85e7, 1200), "Pune": (0.90e7, 1200), "Gurugram": (0.95e7, 1200),
        "Chennai": (0.85e7, 1200), "Kolkata": (0.75e7, 800), "Nagpur": (0.70e7, 1200)}
CENTRAL_FIXED = 1.10e7          # S0: national vendor single large facility (Gautam Buddha Nagar, UP)
PROC_VAR = 6500                 # INR/t processing (reported for completeness; same in all scenarios)
# REAL recycler network: CPCB registry (Aug-2026), aggregated to 107 districts with authorised capacity.
# Contestable share of each district's capacity available to this portfolio = 2% (assumption; never binding at
# 1,075 t/yr onward flow, so the model effectively picks the nearest suitable licensed cluster).
from config import RAW
_reg = pd.read_csv(TAB / "t01_recycler_registry_clean.csv")
_coord = pd.read_csv(RAW / "district_coordinates_approx.csv").set_index("district")
_d = _reg.groupby(["district", "state"]).capacity_tpa.sum().reset_index()
CONTESTABLE = 0.02
RECYCLERS, RSTATE = {}, {}
for r in _d.itertuples():
    key = f"{r.district} ({r.state[:3]})"
    RECYCLERS[key] = r.capacity_tpa * CONTESTABLE
    RSTATE[key] = r.state
    CITY[key] = (_coord.loc[r.district, "lat"], _coord.loc[r.district, "lon"])
NORTH = [k for k, st in RSTATE.items() if st in ("Uttar Pradesh", "Uttarakhand", "Haryana")]
# IT assets are LOW-DENSITY cargo: a 32-ft truck "cubes out" at ~6-7 t, so per-tonne-km rates are well above bulk freight.
# Leg 1 (client -> hub) carries UN-WIPED data-bearing media -> secure chain-of-custody vehicle (GPS lock, sealed, escort).
LTL = 14.0 + 3.0     # INR per t-km: part-load volumetric rate + security premium for data-bearing assets
FTL = 8.0            # INR per t-km: full-truck, sanitised material hub -> recycler
FIRST_MILE = 1800    # INR per t: intra-city collection (identical in all scenarios)
EF_LTL, EF_FTL = 0.20, 0.11   # kg CO2e per t-km (assumed; low-density loads, GLEC-style factors for Indian trucks)

# ---------- recompute rule weight share from synthetic data ----------
assets = pd.read_csv(SYN / "it_asset_returns_synthetic.csv")
rule_mask = (assets.age_yrs >= 4) | (assets.functional_fail == 1)
RULE_RECYCLE_WT = assets.loc[rule_mask, "weight_kg"].sum() / assets.weight_kg.sum()

# ---------- Step A: synthetic sites + K-means ----------
sites = []
for c, t in DEMAND.items():
    n = max(4, int(t / 6))
    lat, lon = CITY[c]
    w = rng.dirichlet(np.ones(n)) * t
    for i in range(n):
        sites.append(dict(site_id=f"{c[:3].upper()}{i:03d}", city=c, lat=lat + rng.normal(0, 0.09),
                          lon=lon + rng.normal(0, 0.09), tonnes=w[i]))
sites = pd.DataFrame(sites); sites.to_csv(SYN / "client_pickup_sites_synthetic.csv", index=False)
Xs = sites[["lat", "lon"]].values
sil = {}
for k in range(4, 11):
    km = KMeans(k, n_init=20, random_state=SEED).fit(Xs, sample_weight=sites.tonnes)
    sil[k] = silhouette_score(Xs, km.labels_)
# Silhouette keeps rising with k (cities are tight point-clouds), so use a parsimony rule:
# stop at the first k where adding one more cluster improves silhouette by < 0.03
best_k = next((k for k in range(4, 10) if sil[k + 1] - sil[k] < 0.03), 6)
km = KMeans(best_k, n_init=20, random_state=SEED).fit(Xs, sample_weight=sites.tonnes)
sites["cluster"] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=["lat", "lon"])
cent["tonnes"] = sites.groupby("cluster").tonnes.sum()
cent["nearest_candidate_hub"] = [min(HUBS, key=lambda h: road_km((r.lat, r.lon), CITY[h])) for r in cent.itertuples()]
cent.to_csv(TAB / "t04_cluster_centroids.csv", index=False)

# ---------- Step B: MILP ----------
def solve(allowed_recyclers):
    s = pywraplp.Solver.CreateSolver("SCIP")
    y = {h: s.BoolVar(f"y_{h}") for h in HUBS}
    x = {(c, h): s.BoolVar(f"x_{c}_{h}") for c in DEMAND for h in HUBS}
    f = {(h, r): s.NumVar(0, s.infinity(), f"f_{h}_{r}") for h in HUBS for r in allowed_recyclers}
    for c in DEMAND: s.Add(sum(x[c, h] for h in HUBS) == 1)
    for h in HUBS:
        load = sum(DEMAND[c] * x[c, h] for c in DEMAND)
        s.Add(load <= HUBS[h][1] * y[h])
        s.Add(sum(f[h, r] for r in allowed_recyclers) == ONWARD * load)
        for c in DEMAND: s.Add(x[c, h] <= y[h])
    for r in allowed_recyclers: s.Add(sum(f[h, r] for h in HUBS) <= RECYCLERS[r])
    leg1 = {(c, h): (0 if c == h else road_km(CITY[c], CITY[h])) for c in DEMAND for h in HUBS}
    leg2 = {(h, r): max(40.0, road_km(CITY[h], CITY[r])) for h in HUBS for r in allowed_recyclers}   # >=40 km intra-district haul
    cost = (sum(HUBS[h][0] * y[h] for h in HUBS)
            + sum(DEMAND[c] * (FIRST_MILE + LTL * leg1[c, h]) * x[c, h] for c in DEMAND for h in HUBS)
            + sum(FTL * leg2[h, r] * f[h, r] for h in HUBS for r in allowed_recyclers))
    s.Minimize(cost)
    status = s.Solve()
    assert status == pywraplp.Solver.OPTIMAL
    open_h = [h for h in HUBS if y[h].solution_value() > 0.5]
    assign = {c: h for c in DEMAND for h in HUBS if x[c, h].solution_value() > 0.5}
    flows = {(h, r): f[h, r].solution_value() for h in HUBS for r in allowed_recyclers if f[h, r].solution_value() > 1e-6}
    tkm_ltl = sum(DEMAND[c] * leg1[c, assign[c]] for c in DEMAND)
    tkm_ftl = sum(v * leg2[k] for k, v in flows.items())
    fixed = sum(HUBS[h][0] for h in open_h)
    transport = s.Objective().Value() - fixed
    return dict(data_tkm=tkm_ltl, cost=s.Objective().Value(), fixed=fixed, transport=transport, open_hubs=open_h, assign=assign,
                flows={f"{k[0]}->{k[1]}": round(v, 1) for k, v in flows.items()},
                tkm=tkm_ltl + tkm_ftl, co2_t=(tkm_ltl * EF_LTL + tkm_ftl * EF_FTL) / 1000,
                avg_haul_onward_km=tkm_ftl / max(1e-9, sum(flows.values())))

# S0: all retired assets travel LTL to the national vendor facility at Gautam Buddha Nagar (recycling happens on site)
tot = sum(DEMAND.values())
d0 = {c: road_km(CITY[c], CITY["Gautam Buddha Nagar (Utt)"]) for c in DEMAND}   # vendor in GB Nagar (real cluster)
tkm0 = sum(DEMAND[c] * d0[c] for c in DEMAND)
cost0 = CENTRAL_FIXED + sum(DEMAND[c] * (FIRST_MILE + LTL * d0[c]) for c in DEMAND)
S0 = dict(cost=cost0, fixed=CENTRAL_FIXED, transport=cost0 - CENTRAL_FIXED, tkm=tkm0, co2_t=tkm0 * EF_LTL / 1000,
          avg_haul_onward_km=tkm0 / tot, open_hubs=["Gautam Buddha Nagar (vendor)"], data_tkm=tkm0)
S1 = solve(NORTH)
S2 = solve(list(RECYCLERS))

# Value-recovery effect (from ML module), scaled to portfolio units
units = tot * 1000 / ml["avg_weight_kg"]
value_uplift = units * ml["uplift_per_asset"]
scen = pd.DataFrame([dict(scenario=n, data_bearing_tkm_mn=d["data_tkm"] / 1e6, total_cost_cr=d["cost"] / 1e7, fixed_cr=d["fixed"] / 1e7,
                          transport_cr=d["transport"] / 1e7, tkm_mn=d["tkm"] / 1e6, co2_t=d["co2_t"],
                          avg_onward_haul_km=d["avg_haul_onward_km"], hubs=", ".join(d["open_hubs"]) or "-")
                     for n, d in [("S0 Status quo", S0), ("S1 Hub & spoke (north recyclers)", S1),
                                  ("S2 Optimised network", S2)]])
scen.to_csv(TAB / "t04_network_scenarios.csv", index=False)

# Fig 4a cluster map + 4b scenario comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
ax = axes[0]
cmap = plt.get_cmap("Dark2")
ax.scatter(sites.lon, sites.lat, s=sites.tonnes * 6, c=[cmap(i) for i in sites.cluster], alpha=0.6, edgecolor="none")
ax.scatter(cent.lon, cent.lat, marker="X", s=160, c=C["ink"], label="K-means centroid")
for h in S2["open_hubs"]:
    ax.scatter(*CITY[h][::-1], marker="s", s=120, facecolor="none", edgecolor=C["accent"], lw=2)
    ax.text(CITY[h][1] - 4.6, CITY[h][0] + 0.2, f"Hub: {h}", fontsize=8, color=C["accent"], weight="bold")
for r in RECYCLERS:
    ax.scatter(*CITY[r][::-1], marker="^", s=max(8, RECYCLERS[r] / CONTESTABLE / 6000), c=C["green"], alpha=0.5)
ax.scatter([], [], marker="^", c=C["green"], label="CPCB-registered recycler district (size = capacity)")
ax.scatter([], [], marker="s", facecolor="none", edgecolor=C["accent"], label="Optimal hub (MILP)")
ax.set_xlim(68.5, 93); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_title(f"Fig 4a. K-means regions (k={best_k}) and MILP hubs", fontsize=11)
ax = axes[1]
xs = np.arange(3)
ax.bar(xs, scen.fixed_cr, color=C["grey"], label="Facility fixed cost")
ax.bar(xs, scen.transport_cr, bottom=scen.fixed_cr, color=C["amber"], label="Collection + freight")
for i, r in scen.iterrows():
    ax.text(i, r.total_cost_cr + 0.08, f"Rs {r.total_cost_cr:.2f} Cr\n{r.co2_t:,.0f} t CO2e", ha="center", fontsize=8)
ax.set_xticks(xs, ["S0\nCentral vendor\n(status quo)", "S1\nHub & spoke\n(north recyclers)", "S2\nOptimised\nnetwork"])
ax.set_ylabel("Annual logistics cost (INR crore)"); ax.set_ylim(0, scen.total_cost_cr.max() * 1.25)
ax.set_title("Fig 4b. Annual cost and CO2e by scenario", fontsize=11); ax.legend(fontsize=8, frameon=False)
fig.savefig(FIG / "fig04_network.png"); plt.close(fig)

res = dict(total_tonnes=tot, units_per_year=round(units), onward_share=ONWARD, rule_recycle_weight_share=round(RULE_RECYCLE_WT, 3),
           silhouette={k: round(v, 3) for k, v in sil.items()}, best_k=best_k,
           centroid_hubs=cent.nearest_candidate_hub.tolist(),
           S0=dict(cost_cr=round(S0["cost"] / 1e7, 2), co2_t=round(S0["co2_t"]), tkm_mn=round(S0["tkm"] / 1e6, 2), haul_km=round(S0["avg_haul_onward_km"])),
           S1=dict(cost_cr=round(S1["cost"] / 1e7, 2), co2_t=round(S1["co2_t"]), tkm_mn=round(S1["tkm"] / 1e6, 2), hubs=S1["open_hubs"], haul_km=round(S1["avg_haul_onward_km"])),
           S2=dict(cost_cr=round(S2["cost"] / 1e7, 2), co2_t=round(S2["co2_t"]), tkm_mn=round(S2["tkm"] / 1e6, 2), hubs=S2["open_hubs"],
                   assign=S2["assign"], flows=S2["flows"], haul_km=round(S2["avg_haul_onward_km"]), fixed_cr=round(S2["fixed"] / 1e7, 2)),
           value_uplift_cr=round(value_uplift / 1e7, 2))
res["S2_vs_S0_cost_saving_pct"] = round(100 * (1 - S2["cost"] / S0["cost"]), 1)
res["S2_vs_S0_co2_saving_pct"] = round(100 * (1 - S2["co2_t"] / S0["co2_t"]), 1)
res["S2_vs_S0_tkm_saving_pct"] = round(100 * (1 - S2["tkm"] / S0["tkm"]), 1)
res["data_tkm_mn"] = {k: round(d["data_tkm"] / 1e6, 2) for k, d in [("S0", S0), ("S1", S1), ("S2", S2)]}
res["S2_vs_S0_data_exposure_cut_pct"] = round(100 * (1 - S2["data_tkm"] / S0["data_tkm"]), 1)
res["S2_vs_S0_cost_saving_cr"] = round((S0["cost"] - S2["cost"]) / 1e7, 2)
save_result("network", res); print(json.dumps(res, indent=1, default=float))

# ---------- Sensitivity: is the S2 design robust to freight-rate and fixed-cost errors? ----------
base = dict(LTL=LTL, FTL=FTL, HUBS=dict(HUBS))
sens = []
for label, ltl_m, fix_m in [("Freight -30%", 0.7, 1.0), ("Freight +30%", 1.3, 1.0),
                            ("Hub fixed cost +30%", 1.0, 1.3), ("Hub fixed cost -30%", 1.0, 0.7),
                            ("Worst case: freight -30%, fixed +30%", 0.7, 1.3)]:
    LTL, FTL = base["LTL"] * ltl_m, base["FTL"] * ltl_m
    HUBS = {h: (v[0] * fix_m, v[1]) for h, v in base["HUBS"].items()}
    s2 = solve(list(RECYCLERS))
    s0 = CENTRAL_FIXED * fix_m + sum(DEMAND[c] * (FIRST_MILE + LTL * d0[c]) for c in DEMAND)
    sens.append(dict(case=label, hubs=", ".join(s2["open_hubs"]), s0_cr=round(s0 / 1e7, 2),
                     s2_cr=round(s2["cost"] / 1e7, 2), saving_pct=round(100 * (1 - s2["cost"] / s0), 1)))
LTL, FTL, HUBS = base["LTL"], base["FTL"], base["HUBS"]
pd.DataFrame(sens).to_csv(TAB / "t04_sensitivity.csv", index=False)
save_result("network_sensitivity", sens)
print(pd.DataFrame(sens).to_string())
