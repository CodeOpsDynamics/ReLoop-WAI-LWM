"""
06 - Risk prioritisation of the ITAD reverse-logistics chain using FUZZY FMEA, with REAL practitioner ratings.
Data: anonymous Google Form survey of practitioners (4-5 Oct 2026), data/raw/fmea_practitioner_survey_oct2026.csv.
Method (Wang et al., 2009; Liu et al., 2013; applied to Indian trucking risk by Dadsena, Sarmah & Naikan, 2019):
  linguistic ratings -> triangular fuzzy numbers (TFN); experience-weighted aggregation; fuzzy RPN = S (x) O (x) D;
  centroid defuzzification; comparison with conventional crisp RPN.
Data-quality rule: a response with zero variance across all 30 ratings ("straight-lining") carries no information
about relative risk and is excluded from the main analysis; results including it are reported as a robustness check.
The earlier LLM-simulated panel is retained only as a benchmark to show how much real judgement changed the ranking.
Concept link: LWM Sessions 10-11 (resilience & transport risk), Session 18 (logistics performance control).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config import RAW, FIG, TAB, SYN, C, save_result

TFN = {"VL": (1, 1, 3), "L": (1, 3, 5), "M": (3, 5, 7), "H": (5, 7, 9), "VH": (7, 9, 10)}
LING = {"Very Low": "VL", "Low": "L", "Medium": "M", "High": "H", "Very High": "VH"}
EXP_W = {"<5": 1, "5–10": 2, "10–15": 3, "15+": 4}          # experience-based expert weights (normalised later)
FM = [("R1", "Data breach: un-wiped drive lost in transit", "Collection & transport"),
      ("R2", "Leakage to informal kabadiwala channel", "Collection"),
      ("R3", "Li-ion battery thermal event in truck/hub", "Transport & storage"),
      ("R4", "EPR / Form-6 manifest non-compliance", "Documentation"),
      ("R5", "Chain-of-custody gap (serial mismatch)", "Hub receiving"),
      ("R6", "Mis-grading destroys recoverable value", "Grading"),
      ("R7", "Return-volume surge (refresh cycle)", "Planning"),
      ("R8", "Recycler capacity / licence lapse", "Recycler"),
      ("R9", "Transit damage lowers resale grade", "Transport & packaging"),
      ("R10", "Commodity price swing on recovered metals", "Market")]
# LLM-simulated benchmark panel used before real data was collected (S, O, D for 3 personas)
SIM = {"R1": (["VH","VH","VH"], ["L","M","L"], ["H","VH","H"]), "R2": (["H","H","VH"], ["M","M","H"], ["H","H","VH"]),
       "R3": (["VH","H","VH"], ["L","L","M"], ["M","M","H"]), "R4": (["H","M","VH"], ["M","L","M"], ["M","M","M"]),
       "R5": (["H","VH","H"], ["H","M","H"], ["M","H","M"]), "R6": (["M","L","L"], ["H","M","M"], ["H","M","H"]),
       "R7": (["M","L","L"], ["M","M","M"], ["L","L","M"]), "R8": (["H","M","H"], ["L","L","M"], ["M","L","M"]),
       "R9": (["M","L","L"], ["H","M","M"], ["L","L","M"]), "R10": (["M","L","M"], ["M","M","M"], ["VL","L","VL"])}

# ---------------- load and clean the survey ----------------
raw = pd.read_csv(RAW / "fmea_practitioner_survey_oct2026.csv")
rate_cols = [c for c in raw.columns if c.endswith("[Severity]") or c.endswith("[Occurrence]") or c.endswith("[Detection]")]
assert len(rate_cols) == 30
resp = []
for i, r in raw.iterrows():
    codes = [LING[r[c].strip()] for c in rate_cols]
    mids = [TFN[c][1] for c in codes]
    resp.append(dict(expert=f"P{i+1}", role=r["Your role:"], experience=r["Years of experience:"],
                     consent=r["Can I mention your role (not name) in my report?"], codes=codes,
                     sd=float(np.std(mids)), n_distinct=len(set(codes))))
panel = pd.DataFrame(resp)
panel["straight_lined"] = panel.sd == 0
panel["weight_raw"] = panel.experience.map(EXP_W)
panel[["expert", "role", "experience", "consent", "n_distinct", "sd", "straight_lined", "weight_raw"]].to_csv(TAB / "t06_panel_quality.csv", index=False)

def ratings_for(sub):
    """Return {Rk: (S list, O list, D list)} and normalised weights for a subset of experts."""
    out = {}
    for k, (fid, _, _) in enumerate(FM):
        s, o, d = [], [], []
        for codes in sub.codes:
            s.append(codes[3 * k]); o.append(codes[3 * k + 1]); d.append(codes[3 * k + 2])
        out[fid] = (s, o, d)
    w = sub.weight_raw.values.astype(float); return out, w / w.sum()

def fmea(ratings, w):
    def agg(lst):
        m = np.array([TFN[x] for x in lst], dtype=float); return tuple((w[:, None] * m).sum(axis=0))
    rows = []
    for fid, name, stage in FM:
        S, O, D = ratings[fid]
        s, o, d = agg(S), agg(O), agg(D)
        fr = (s[0] * o[0] * d[0], s[1] * o[1] * d[1], s[2] * o[2] * d[2])
        crisp = np.mean([TFN[x][1] for x in S]) * np.mean([TFN[x][1] for x in O]) * np.mean([TFN[x][1] for x in D])
        rows.append(dict(id=fid, failure_mode=name, stage=stage, S=round(sum(s) / 3, 2), O=round(sum(o) / 3, 2), D=round(sum(d) / 3, 2),
                         FRPN_low=fr[0], FRPN_high=fr[2], FRPN=round(sum(fr) / 3, 1), crisp_RPN=round(crisp, 1)))
    df = pd.DataFrame(rows)
    df["fuzzy_rank"] = df.FRPN.rank(ascending=False, method="min").astype(int)
    df["crisp_rank"] = df.crisp_RPN.rank(ascending=False, method="min").astype(int)
    return df.sort_values(["fuzzy_rank", "id"])

main_panel = panel[~panel.straight_lined]
R_main, w_main = ratings_for(main_panel)
R_all, w_all = ratings_for(panel)
df = fmea(R_main, w_main)
df_all = fmea(R_all, w_all)
df_sim = fmea(SIM, np.array([0.40, 0.35, 0.25]))
df.to_csv(TAB / "t06_fuzzy_fmea.csv", index=False)
df_all.to_csv(TAB / "t06_fuzzy_fmea_all4_robustness.csv", index=False)
df_sim.to_csv(TAB / "t06_fuzzy_fmea_simulated_benchmark.csv", index=False)

# Robustness to the weighting scheme: equal weights, and one 15+ expert re-coded as 10-15 years
df_eq = fmea(R_main, np.ones(len(main_panel)) / len(main_panel))
_w_alt = main_panel.weight_raw.values.astype(float); _w_alt[np.argmax(_w_alt)] = EXP_W["10–15"]
df_alt = fmea(R_main, _w_alt / _w_alt.sum())
rho_eq = spearmanr(df.set_index("id").fuzzy_rank.sort_index(), df_eq.set_index("id").fuzzy_rank.sort_index()).correlation
rho_alt = spearmanr(df.set_index("id").fuzzy_rank.sort_index(), df_alt.set_index("id").fuzzy_rank.sort_index()).correlation
cmp_ = df[["id", "fuzzy_rank"]].merge(df_all[["id", "fuzzy_rank"]], on="id", suffixes=("", "_all4")).merge(
       df_sim[["id", "fuzzy_rank"]].rename(columns={"fuzzy_rank": "rank_simulated"}), on="id")
rho_all = spearmanr(cmp_.fuzzy_rank, cmp_.fuzzy_rank_all4).correlation
rho_sim = spearmanr(cmp_.fuzzy_rank, cmp_.rank_simulated).correlation
rho_crisp = spearmanr(df.fuzzy_rank, df.crisp_rank).correlation
cmp_.to_csv(TAB / "t06_rank_comparison.csv", index=False)

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axes[0]
y = np.arange(len(df))[::-1]
ax.hlines(y, df.FRPN_low, df.FRPN_high, color=C["light"], lw=6, label="Fuzzy RPN spread (low-high)")
ax.scatter(df.FRPN, y, color=C["dark"], zorder=3, label="Defuzzified FRPN (centroid)")
ax.scatter(df.crisp_RPN, y, marker="x", color=C["accent"], zorder=4, label="Conventional crisp RPN")
hi_sev = df.S >= 7
ax.set_yticks(y, [f"{a}  {b[:46]}" + ("  [S>=7]" if h else "") for a, b, h in zip(df.id, df.failure_mode, hi_sev)], fontsize=8)
ax.set_xlabel("Risk priority number"); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
ax.set_title(f"Fig 6a. Fuzzy FMEA, practitioner panel (n={len(main_panel)})", fontsize=11)
ax = axes[1]
order = df.id.tolist()
sim_rank = cmp_.set_index("id").rank_simulated
real_rank = cmp_.set_index("id").fuzzy_rank
for fid in order:
    col = C["accent"] if abs(sim_rank[fid] - real_rank[fid]) >= 3 else C["grey"]
    ax.plot([0, 1], [sim_rank[fid], real_rank[fid]], "-o", color=col, lw=1.4)
    ax.text(-0.05, sim_rank[fid], fid, ha="right", va="center", fontsize=8)
    ax.text(1.05, real_rank[fid], fid, ha="left", va="center", fontsize=8)
ax.set_xticks([0, 1], ["Simulated\n(LLM panel)", "Real\npractitioners"]); ax.set_ylim(10.7, 0.3); ax.set_xlim(-0.3, 1.3)
ax.set_ylabel("Rank (1 = highest risk)"); ax.grid(False)
ax.set_title(f"Fig 6b. Rank shift (Spearman rho = {rho_sim:.2f})", fontsize=11)
fig.savefig(FIG / "fig06_fuzzy_fmea.png"); plt.close(fig)

roles = main_panel.role.value_counts().to_dict()
res = dict(n_responses=len(panel), n_used=len(main_panel), excluded=panel[panel.straight_lined].expert.tolist(),
           excluded_reason="all 30 ratings identical (Medium): straight-lining",
           roles_used=roles, experience_used=main_panel.experience.tolist(),
           weights_used=dict(zip(main_panel.expert, np.round(w_main, 3))),
           ranking=df[["id", "failure_mode", "FRPN", "crisp_RPN", "fuzzy_rank", "crisp_rank", "S", "O", "D"]].to_dict("records"),
           top3=df.head(3).id.tolist(), top3_all4=df_all.head(3).id.tolist(), top3_simulated=df_sim.head(3).id.tolist(),
           rho_real_vs_all4=round(rho_all, 2), rho_real_vs_simulated=round(rho_sim, 2), rho_fuzzy_vs_crisp=round(rho_crisp, 2),
           severity_ge7=df[df.S >= 7].id.tolist(),
           top3_equal_weights=df_eq.head(3).id.tolist(), rho_equal_weights=round(rho_eq, 2),
           top3_alt_weights=df_alt.head(3).id.tolist(), rho_alt_weights=round(rho_alt, 2),
           panel_roles_described="IT asset management; aviation engineering; consulting data analytics",
           rank_changes=[dict(id=r.id, simulated=int(r.rank_simulated), real=int(r.fuzzy_rank)) for r in cmp_.itertuples() if r.rank_simulated != r.fuzzy_rank],
           other_risks_text=[str(x) for x in raw.iloc[:, -1].dropna().tolist()])
save_result("fmea", res)
import json; print(json.dumps({k: v for k, v in res.items() if k != "ranking"}, indent=1, default=float))
print(df[["id", "failure_mode", "S", "O", "D", "FRPN", "crisp_RPN", "fuzzy_rank", "crisp_rank"]].to_string())
