"""
01b - REAL-DATA unsupervised ML: typology of state e-waste reverse-logistics systems.
Unit: TRAI service area (16 areas, ~95% of India's population). Features are SIZE-NEUTRAL ratios so that clusters
reflect the *role* a region plays, not merely how big it is:
  MI  = share of formal processing / share of generation (internet-subscriber proxy)
  CI  = share of authorised capacity / share of generation
  UT  = FY25 processed / authorised capacity
K-means (standardised), k chosen by silhouette over 2-5. Sources: CPCB registry (Aug-2026), PIB FY25, TRAI/Census.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import TAB, FIG, C, SEED, save_result

a = pd.read_csv(TAB / "t01_mismatch_two_proxies.csv")
a["CI"] = a.cap_share / a.gen_share_internet
a["UT"] = a.utilisation_fy25.fillna(0)
feats = pd.DataFrame({"log MI": np.log(a.MI_internet + 0.02), "log CI": np.log(a.CI + 0.02), "UT": a.UT})
X = StandardScaler().fit_transform(feats)
sil = {k: silhouette_score(X, KMeans(k, n_init=100, random_state=SEED).fit_predict(X)) for k in range(2, 6)}
k = max(sil, key=sil.get)
a["cluster"] = KMeans(k, n_init=100, random_state=SEED).fit_predict(X)
prof = a.groupby("cluster").agg(n=("label", "count"), MI=("MI_internet", "median"), CI=("CI", "median"),
                                UT=("UT", "median"), gen=("gen_share_internet", "sum"), proc=("proc_share", "sum"),
                                cap=("cap_share", "sum")).reset_index()
def name(r):
    if r.MI >= 1: return "Processing hubs (net importers)"
    if r.CI < 0.05: return "Processing deserts"
    if r.UT >= 0.5: return "Small, saturated"
    return "Under-fed capacity (net exporters)"
prof["segment"] = prof.apply(name, axis=1)
a["segment"] = a.cluster.map(dict(zip(prof.cluster, prof.segment)))
a[["label", "segment", "MI_internet", "CI", "UT", "gen_share_internet", "proc_share", "cap_share"]].sort_values(
    ["segment", "MI_internet"], ascending=[True, False]).round(3).to_csv(TAB / "t01b_state_typology.csv", index=False)

fig, ax = plt.subplots(figsize=(8.6, 5.2))
col = {"Processing hubs (net importers)": C["dark"], "Under-fed capacity (net exporters)": C["amber"], "Processing deserts": C["accent"], "Small, saturated": C["mid"]}
FLOOR = 0.003
a["x"] = a.CI.clip(lower=FLOOR); a["y"] = a.MI_internet.clip(lower=FLOOR)
for s_, g in a.groupby("segment"):
    ax.scatter(g.x, g.y, s=g.gen_share_internet * 2500, color=col.get(s_, C["grey"]), alpha=0.85, label=f"{s_} ({len(g)})", edgecolor="white")
MAN = {"West Bengal": (-62, -12), "Punjab": (6, 4), "Haryana": (-10, 8), "UP + Uttarakhand": (-110, 6), "Kerala": (-38, -12),
       "Madhya Pradesh": (-30, 8), "Chhattisgarh": (4, -12), "Rajasthan": (-58, -3), "Karnataka": (8, 6), "Tamil Nadu": (8, -12),
       "Maharashtra": (-82, 12), "AP + Telangana": (8, 6), "Gujarat": (8, -10), "Bihar": (8, 4), "Odisha": (8, -12), "Jharkhand": (-60, 6)}
for r in a.itertuples():
    ax.annotate(r.label, (r.x, r.y), fontsize=7.5, xytext=MAN.get(r.label, (6, 4)), textcoords="offset points")
ax.axhline(1, color=C["grey"], ls=":", lw=1); ax.axvline(1, color=C["grey"], ls=":", lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(FLOOR / 2, 8); ax.set_ylim(FLOOR / 2, 8)
ax.set_xlabel("Capacity index CI = capacity share / generation share (log; zero shown at floor)")
ax.set_ylabel("Mismatch index MI = processing share / generation share (log)")
ax.set_title(f"Fig 1e. K-means typology of regional e-waste systems (k={k}, silhouette {sil[k]:.2f})", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(FIG / "fig01e_state_typology.png"); plt.close(fig)

res = dict(k=k, silhouette={kk: round(v, 3) for kk, v in sil.items()},
           segments={s: g.label.tolist() for s, g in a.groupby("segment")},
           profile=prof.drop(columns="cluster").round(3).to_dict("records"))
save_result("typology", res)
import json; print(json.dumps(res, indent=1, default=float))
