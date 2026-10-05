"""
01 - REAL-DATA diagnostics of India's formal e-waste reverse-logistics system.
Sources (all public, in data/raw):
  * CPCB EPR portal "Recycler Registration Granted List" (downloaded 23-Aug-2026): 413 rows -> 411 after cleaning
  * Lok Sabha USQ 983 (18-Sep-2020) and USQ 2458 (13-Dec-2021): recyclers, capacity, collection centres, processing
  * PIB release 2147876 (24-Jul-2025): state processing FY24, FY25
  * TRAI tele-/internet density by state, Sep-2024 (Jharkhand Economic Survey 2025-26, Table 9.8)
  * Census of India 2011 population
Outputs: data-cleaning log, mismatch index (two independent generation proxies), capacity atlas, utilisation,
district concentration.  Concept link: LWM Sessions 7-9 (green logistics, freight), 12-13 (facility location).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from config import RAW, FIG, TAB, C, save_result

# ---------------- 1. Clean the recycler registry ----------------
reg = pd.read_csv(RAW / "cpcb_recycler_registry_raw_aug2026.csv")
log = {"rows_raw": len(reg)}
test_rows = reg.company.str.contains("Recycler Test", case=False)
log["removed_test_records"] = int(test_rows.sum())
reg = reg[~test_rows].copy()
dup = reg.duplicated(subset=["company", "address"])
log["removed_exact_duplicates"] = int(dup.sum()); reg = reg[~dup]
log["rows_clean"] = len(reg); log["districts"] = int(reg.district.nunique()); log["states"] = int(reg.state.nunique())
log["total_capacity_tpa"] = float(reg.capacity_tpa.sum())
log["median_capacity_tpa"] = float(reg.capacity_tpa.median())
log["share_units_under_1000tpa"] = round(float((reg.capacity_tpa < 1000).mean()), 3)
reg.to_csv(TAB / "t01_recycler_registry_clean.csv", index=False)

# ---------------- 2. State panel ----------------
pib = pd.read_csv(RAW / "pib_state_processing.csv")
ls20 = pd.read_csv(RAW / "ls_2020_q983_recyclers_capacity.csv")
ls21 = pd.read_csv(RAW / "ls_2021_q2458_state_annexures.csv")
cap26 = reg.groupby("state").agg(recyclers_2026=("company", "count"), capacity_tpa_2026=("capacity_tpa", "sum")).reset_index()
panel = (ls21.merge(ls20, on="state", how="outer").merge(pib, on="state", how="outer").merge(cap26, on="state", how="outer"))
panel.to_csv(TAB / "t01_state_panel.csv", index=False)

# ---------------- 3. Generation proxies at TRAI service-area level ----------------
# TRAI service areas: "Andhra Pradesh" covers AP + Telangana; "Uttar Pradesh" covers UP + Uttarakhand (East + West LSAs)
GROUP = {"Telangana": "Andhra Pradesh", "Uttarakhand": "Uttar Pradesh"}
LABEL = {"Andhra Pradesh": "AP + Telangana", "Uttar Pradesh": "UP + Uttarakhand"}
pop = pd.read_csv(RAW / "census2011_population.csv"); pop["area"] = pop.state.replace(GROUP)
trai = pd.read_csv(RAW / "trai_ict_density_sep2024_jes.csv")
area = pop.groupby("area").population_2011.sum().reset_index().merge(trai, left_on="area", right_on="trai_area")
area["internet_subs_idx"] = area.internet_density / 100 * area.population_2011      # device-stock proxy
gsdp = pd.read_csv(RAW / "state_gsdp_proxy.csv"); gsdp["area"] = gsdp.state.replace(GROUP)
area = area.merge(gsdp.groupby("area").gsdp_lakh_cr_fy24_approx.sum().reset_index(), on="area", how="left")
p25 = pib.assign(area=pib.state.replace(GROUP)).groupby("area").processed_fy25_t.sum()
c26 = cap26.assign(area=cap26.state.replace(GROUP)).groupby("area").capacity_tpa_2026.sum()
area["processed_fy25_t"] = area.area.map(p25).fillna(0)
area["capacity_tpa_2026"] = area.area.map(c26).fillna(0)
area["gen_share_internet"] = area.internet_subs_idx / area.internet_subs_idx.sum()
area["gen_share_gsdp"] = area.gsdp_lakh_cr_fy24_approx / area.gsdp_lakh_cr_fy24_approx.sum()
area["proc_share"] = area.processed_fy25_t / area.processed_fy25_t.sum()
area["cap_share"] = area.capacity_tpa_2026 / area.capacity_tpa_2026.sum()
area["MI_internet"] = area.proc_share / area.gen_share_internet
area["MI_gsdp"] = area.proc_share / area.gen_share_gsdp
area["utilisation_fy25"] = np.where(area.capacity_tpa_2026 > 0, area.processed_fy25_t / area.capacity_tpa_2026, np.nan)
area["label"] = area.area.replace(LABEL)
area = area.sort_values("gen_share_internet", ascending=False)
area.to_csv(TAB / "t01_mismatch_two_proxies.csv", index=False)
rank_corr = area[["MI_internet", "MI_gsdp"]].rank().corr().iloc[0, 1]

# ---------------- 4. Concentration ----------------
dist = reg.groupby(["state", "district"]).capacity_tpa.sum().sort_values(ascending=False).reset_index()
dist["cum_share"] = dist.capacity_tpa.cumsum() / dist.capacity_tpa.sum()
dist.to_csv(TAB / "t01_district_capacity.csv", index=False)
top10_share = dist.head(10).capacity_tpa.sum() / dist.capacity_tpa.sum()
north = ["Uttar Pradesh", "Uttarakhand", "Haryana"]
north_cap_share = cap26[cap26.state.isin(north)].capacity_tpa_2026.sum() / cap26.capacity_tpa_2026.sum()
north_proc_share = pib[pib.state.isin(north)].processed_fy25_t.sum() / pib.processed_fy25_t.sum()
hhi_cap_state = ((cap26.capacity_tpa_2026 / cap26.capacity_tpa_2026.sum()) ** 2).sum() * 1e4
nat_util = pib.processed_fy25_t.sum() / reg.capacity_tpa.sum()
sw = ["Maharashtra", "Karnataka", "Tamil Nadu", "Telangana", "Gujarat", "Kerala", "Andhra Pradesh"]
sw_cap = cap26[cap26.state.isin(sw)].capacity_tpa_2026.sum()
sw_proc = pib[pib.state.isin(sw)].processed_fy25_t.sum()

# ---------------- Figures ----------------
# Fig 1: mismatch with two proxies
fig, ax = plt.subplots(figsize=(8.6, 5.6))
pl = area.sort_values("gen_share_internet")
y = np.arange(len(pl))
ax.barh(y - 0.2, pl.gen_share_internet * 100, 0.4, color=C["amber"], label="Generation proxy: internet subscribers (TRAI x Census)")
ax.barh(y + 0.2, pl.proc_share * 100, 0.4, color=C["green"], label="Formal processing FY25 (CPCB)")
ax.scatter(pl.gen_share_gsdp * 100, y - 0.2, marker="|", s=120, color=C["ink"], zorder=4, label="Robustness: GSDP-share proxy")
for i, r in enumerate(pl.itertuples()):
    ax.text(max(r.gen_share_internet, r.proc_share) * 100 + 0.8, i, f"MI {r.MI_internet:.2f}", va="center", fontsize=8,
            color=C["accent"] if r.MI_internet < 0.5 else C["dark"])
ax.set_yticks(y, pl.label); ax.set_xlabel("% of national total")
ax.set_title("Fig 1. Where e-waste is generated vs. formally processed (FY 2024-25)")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.savefig(FIG / "fig01_spatial_mismatch.png"); plt.close(fig)

# Fig 1b: capacity atlas
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[0.8, 1.2], wspace=0.45))
ax = axes[0]
yrs = ["Sep 2020\n(LS Q983)", "Dec 2021\n(LS Q2458)", "Aug 2026\n(CPCB registry)"]
capv = [ls20.capacity_tpa_2020.sum() / 1e5, ls21.capacity_tpa_2021.sum() / 1e5, reg.capacity_tpa.sum() / 1e5]
b = ax.bar(yrs, capv, color=[C["light"], C["mid"], C["green"]])
for r_, v, n in zip(b, capv, [ls20.recyclers_2020.sum(), ls21.recyclers_2021.sum(), len(reg)]):
    ax.text(r_.get_x() + r_.get_width() / 2, max(v, 12.5) + 0.6, f"{v:.1f} lakh t\n{n} units", ha="center", va="bottom", fontsize=8)
ax.axhline(pib.processed_fy25_t.sum() / 1e5, color=C["accent"], ls="--", lw=1.2)
ax.text(-0.4, pib.processed_fy25_t.sum() / 1e5 - 2.2, f"FY25 actually processed: {pib.processed_fy25_t.sum()/1e5:.1f} lakh t", color=C["accent"], fontsize=8)
ax.set_ylabel("Authorised capacity (lakh t / yr)"); ax.set_ylim(0, max(capv) * 1.25)
ax.set_title("Fig 1b. Capacity grew 5x; flow did not keep up", fontsize=11)
ax = axes[1]
u = area[area.capacity_tpa_2026 > 5000].sort_values("utilisation_fy25")
ax.barh(u.label, u.utilisation_fy25 * 100, color=[C["accent"] if v < 0.15 else C["green"] for v in u.utilisation_fy25])
for i, r in enumerate(u.itertuples()):
    ax.text(r.utilisation_fy25 * 100 + 1, i, f"{r.utilisation_fy25*100:.0f}%  (cap {r.capacity_tpa_2026/1e5:.1f} lakh t)", va="center", fontsize=7.5)
ax.axvline(nat_util * 100, color=C["grey"], ls="--", lw=1); ax.text(nat_util * 100 + 1, -0.9, f"India {nat_util*100:.0f}%", fontsize=8, color=C["grey"])
ax.set_xlabel("FY25 processed / authorised capacity (%)"); ax.set_xlim(0, 105)
ax.set_title("Fig 1c. Capacity utilisation by service area", fontsize=11)
fig.savefig(FIG / "fig01b_capacity_atlas.png"); plt.close(fig)

# Fig 1d: district concentration
fig, ax = plt.subplots(figsize=(8.6, 4.2))
t15 = dist.head(15)
ax.bar(range(15), t15.capacity_tpa / 1e5, color=[C["dark"] if s in north else C["mid"] for s in t15.state])
ax.set_xticks(range(15), [f"{d}\n({s[:3]})" for d, s in zip(t15.district, t15.state)], fontsize=7, rotation=45, ha="right")
ax2 = ax.twinx(); ax2.plot(range(15), t15.cum_share * 100, "-o", color=C["accent"], ms=3); ax2.set_ylim(0, 100)
ax2.set_ylabel("Cumulative share of national capacity (%)", color=C["accent"]); ax2.grid(False)
ax.set_ylabel("Capacity (lakh t / yr)")
ax.set_title(f"Fig 1d. Ten districts hold {top10_share*100:.0f}% of India's authorised e-waste capacity (dark = UP/UK/HR)", fontsize=10.5)
fig.savefig(FIG / "fig01d_district_concentration.png"); plt.close(fig)

mi = area.set_index("label")
res = dict(cleaning=log, n_recyclers=len(reg), capacity_lakh_t=round(reg.capacity_tpa.sum() / 1e5, 2),
           capacity_2020_lakh_t=round(capv[0], 2), capacity_2021_lakh_t=round(capv[1], 2), capacity_growth_x=round(capv[2] / capv[0], 1),
           recyclers_2020=int(ls20.recyclers_2020.sum()), recyclers_2021=int(ls21.recyclers_2021.sum()),
           national_utilisation_pct=round(nat_util * 100, 1),
           north_belt_share_pct=round(north_proc_share * 100, 1), north_cap_share_pct=round(north_cap_share * 100, 1),
           hhi_capacity_state=round(hhi_cap_state), top10_district_share_pct=round(top10_share * 100, 1),
           top3_districts=dist.head(3).district.tolist(),
           sw_capacity_lakh_t=round(sw_cap / 1e5, 2), sw_processed_lakh_t=round(sw_proc / 1e5, 2), sw_utilisation_pct=round(100 * sw_proc / sw_cap, 1),
           mi_internet=mi.MI_internet.round(2).to_dict(), mi_gsdp=mi.MI_gsdp.round(2).to_dict(),
           util=mi.utilisation_fy25.dropna().round(3).to_dict(), mi_rank_corr=round(rank_corr, 2),
           sw_gen_share_internet_pct=round(100 * area[area.area.isin(["Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Gujarat", "Kerala"])].gen_share_internet.sum(), 1),
           sw_proc_share_pct=round(100 * area[area.area.isin(["Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Gujarat", "Kerala"])].proc_share.sum(), 1),
           up_share_pct=round(100 * pib.set_index("state").processed_fy25_t["Uttar Pradesh"] / pib.processed_fy25_t.sum(), 1),
           recyclers_growth_pct=round(100 * (446 / 291 - 1), 1))
save_result("spatial_mismatch", res)
import json; print(json.dumps(res, indent=1, default=float))
