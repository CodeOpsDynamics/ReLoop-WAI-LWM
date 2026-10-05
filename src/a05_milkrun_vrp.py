"""
05 - First-mile milk-run routing in Pune (capacitated vehicle routing problem, Google OR-Tools).
Problem: client offices raise ad-hoc pickup tickets; today each ticket triggers a dedicated round trip from the hub.
Solution: batch weekly tickets and route secure vans as milk-runs (CVRP with capacity + max route length).
Concept link: LWM Session 4 (route optimisation, XDel case), Session 9 (vehicle utilisation, freight cost),
Sessions 7-8 (CO2 per km).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from config import SYN, FIG, TAB, C, SEED, haversine_km, save_result

rng = np.random.default_rng(SEED)
DEPOT = ("ReLoop Hub - Chakan MIDC", 18.760, 73.860)
SITES = [("Hinjewadi Ph-1", 18.591, 73.738), ("Hinjewadi Ph-3", 18.585, 73.690), ("Wakad", 18.599, 73.762),
         ("Baner", 18.559, 73.786), ("Aundh", 18.558, 73.807), ("Shivajinagar", 18.531, 73.847),
         ("Kalyani Nagar", 18.548, 73.902), ("Viman Nagar", 18.567, 73.914), ("Kharadi EON", 18.551, 73.950),
         ("Magarpatta", 18.514, 73.926), ("Hadapsar", 18.502, 73.939), ("Koregaon Park", 18.536, 73.893),
         ("Yerawada Commerzone", 18.556, 73.885), ("Kothrud", 18.507, 73.807), ("Bavdhan", 18.515, 73.776),
         ("Pimpri", 18.627, 73.800), ("Bhosari MIDC", 18.632, 73.849), ("Talawade IT Park", 18.678, 73.795),
         ("Swargate", 18.501, 73.858), ("Balewadi High St", 18.571, 73.773)]
CIRC = 1.45                         # urban road circuity
VAN_CAP_KG, N_VANS, MAX_ROUTE_KM = 1800, 6, 140
EF_VAN = 0.27                       # kg CO2e per km (diesel LCV, assumed)
COST_PER_KM = 28                    # INR per km (secure LCV incl. driver, assumed)

pts = [DEPOT] + SITES
demand = [0] + list(rng.integers(150, 700, len(SITES)))       # kg ready for pickup this week
n = len(pts)
D = np.array([[0 if i == j else int(round(haversine_km(pts[i][1], pts[i][2], pts[j][1], pts[j][2]) * CIRC * 10))
               for j in range(n)] for i in range(n)])          # deci-km for integer solver

mgr = pywrapcp.RoutingIndexManager(n, N_VANS, 0)
rt = pywrapcp.RoutingModel(mgr)
cb = rt.RegisterTransitCallback(lambda a, b: int(D[mgr.IndexToNode(a), mgr.IndexToNode(b)]))
rt.SetArcCostEvaluatorOfAllVehicles(cb)
dcb = rt.RegisterUnaryTransitCallback(lambda a: int(demand[mgr.IndexToNode(a)]))
rt.AddDimensionWithVehicleCapacity(dcb, 0, [VAN_CAP_KG] * N_VANS, True, "Load")
rt.AddDimension(cb, 0, MAX_ROUTE_KM * 10, True, "Dist")
rt.GetDimensionOrDie("Dist").SetGlobalSpanCostCoefficient(10)
p = pywrapcp.DefaultRoutingSearchParameters()
p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
p.time_limit.seconds = 15
sol = rt.SolveWithParameters(p)

routes, total = [], 0
for v in range(N_VANS):
    idx, r, dist, load = rt.Start(v), [], 0, 0
    while not rt.IsEnd(idx):
        node = mgr.IndexToNode(idx); r.append(node); load += demand[node]
        nxt = sol.Value(rt.NextVar(idx)); dist += D[mgr.IndexToNode(idx), mgr.IndexToNode(nxt)]; idx = nxt
    r.append(0)
    if len(r) > 2:
        routes.append(dict(van=v + 1, stops=" > ".join(pts[i][0] for i in r[1:-1]), n_stops=len(r) - 2,
                           km=dist / 10, load_kg=load, utilisation_pct=round(100 * load / VAN_CAP_KG, 1), path=r))
        total += dist / 10
baseline_km = sum(2 * D[0, i] / 10 for i in range(1, n))     # one dedicated round trip per ticket
baseline_util = np.mean(demand[1:]) / VAN_CAP_KG * 100
rdf = pd.DataFrame(routes); rdf.drop(columns="path").to_csv(TAB / "t05_milkrun_routes.csv", index=False)
pd.DataFrame(dict(site=[p[0] for p in pts], lat=[p[1] for p in pts], lon=[p[2] for p in pts], demand_kg=demand)).to_csv(
    SYN / "pune_pickup_week_synthetic.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axes[0]; cols = plt.get_cmap("Dark2")
for k, r in enumerate(routes):
    xs = [pts[i][2] for i in r["path"]]; ys = [pts[i][1] for i in r["path"]]
    ax.plot(xs, ys, "-o", color=cols(k), ms=4, lw=1.6, label=f"Van {r['van']}: {r['km']:.0f} km, {r['utilisation_pct']:.0f}% full")
ax.scatter(DEPOT[2], DEPOT[1], marker="s", s=120, c=C["accent"], zorder=5); ax.text(DEPOT[2] + 0.005, DEPOT[1], "Hub (Chakan)", fontsize=8)
ax.set_title("Fig 5a. Optimised weekly milk-runs, Pune (CVRP)"); ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax = axes[1]
vals = [baseline_km, total]
b = ax.bar(["Ad-hoc trip\nper ticket", "Optimised\nmilk-run"], vals, color=[C["amber"], C["green"]])
for r_, v in zip(b, vals): ax.text(r_.get_x() + r_.get_width() / 2, v + 8, f"{v:,.0f} km", ha="center")
ax.set_ylabel("Vehicle-km per week"); ax.set_title("Fig 5b. Weekly vehicle-km")
fig.savefig(FIG / "fig05_milkrun.png"); plt.close(fig)

res = dict(sites=len(SITES), total_kg=int(sum(demand)), vans_used=len(routes), baseline_km=round(baseline_km),
           optimised_km=round(total), km_saving_pct=round(100 * (1 - total / baseline_km), 1),
           baseline_util_pct=round(baseline_util, 1), avg_util_pct=round(rdf.utilisation_pct.mean(), 1),
           weekly_cost_saving_inr=round((baseline_km - total) * COST_PER_KM),
           annual_cost_saving_lakh=round((baseline_km - total) * COST_PER_KM * 50 / 1e5, 2),
           annual_co2_saving_t=round((baseline_km - total) * EF_VAN * 50 / 1000, 1),
           routes=rdf.drop(columns="path").to_dict("records"))
save_result("milkrun", res); print({k: v for k, v in res.items() if k != "routes"})
