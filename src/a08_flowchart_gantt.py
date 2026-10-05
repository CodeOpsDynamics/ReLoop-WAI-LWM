"""08 - Methodology flowchart and implementation Gantt chart (figures for report & deck)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from config import FIG, C

# ---------- Methodology flowchart ----------
fig, ax = plt.subplots(figsize=(12, 5.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 5.2)
cols = [("1. Frame", ["CPCB / PIB data", "LWM session map", "GenAI scoping"], C["grey"]),
        ("2. Data", ["Real: CPCB state", "+ national series", "Synthetic (disclosed):", "assets, sites, panel"], C["amber"]),
        ("3. Diagnose", ["Spatial mismatch", "index", "Logistic + policy", "forecast"], C["mid"]),
        ("4. Predict", ["Random Forest", "asset grading", "K-means site", "regions"], C["green"]),
        ("5. Optimise", ["MILP hub location", "(OR-Tools SCIP)", "CVRP milk-runs"], C["green"]),
        ("6. Risk & value", ["Fuzzy FMEA", "NPV + scenarios", "Sensitivity"], C["dark"]),
        ("7. Decide", ["Recommendations", "Roadmap", "Dashboard"], C["accent"])]
w, gap = 1.5, 0.18
for i, (title, items, col) in enumerate(cols):
    x = 0.1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 1.3), w, 3.0, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, ec="none"))
    ax.text(x + w / 2, 3.9, title, ha="center", va="center", color="white", fontsize=10, weight="bold")
    for j, it in enumerate(items):
        ax.text(x + w / 2, 3.25 - j * 0.42, it, ha="center", va="center", color="white", fontsize=8)
    if i < len(cols) - 1:
        ax.annotate("", xy=(x + w + gap - 0.01, 2.8), xytext=(x + w + 0.01, 2.8),
                    arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.2))
ax.text(6, 0.75, "Human-in-the-loop: every AI output (code, numbers, citations) verified by the author; all prompts logged",
        ha="center", fontsize=9, style="italic", color=C["ink"])
ax.set_title("Fig M. ReLoop methodology: AI-augmented analytics pipeline", fontsize=12, weight="bold")
fig.savefig(FIG / "figM_methodology.png"); plt.close(fig)

# ---------- Gantt ----------
tasks = [("Phase 0: Business case sign-off, data access, FMEA re-run with hub staff", 0, 1, C["grey"]),
         ("Phase 1: Pilot hub - Pune (Chakan) + Pune milk-runs", 1, 4, C["mid"]),
         ("Phase 1: AI grading model on real asset register (shadow mode)", 1, 5, C["mid"]),
         ("Phase 1: Secure-van IoT locks; chain-of-custody SOP", 2, 4, C["mid"]),
         ("Phase 2: Bengaluru hub go-live; regional recycler contracts", 5, 8, C["green"]),
         ("Phase 2: AI grading live with human override (>=95% audit)", 5, 9, C["green"]),
         ("Phase 2: Control-tower dashboard; EPR auto-reporting", 6, 9, C["green"]),
         ("Phase 3: Gurugram hub; full network; R2v3 / ISO 27001 audit", 9, 12, C["dark"]),
         ("Phase 3: Model retraining loop; scale to new clients", 10, 15, C["dark"]),
         ("Change management & training (continuous)", 0, 15, C["amber"])]
fig, ax = plt.subplots(figsize=(11, 5))
for i, (t, s0, e, col) in enumerate(tasks[::-1]):
    ax.barh(i, e - s0, left=s0, color=col, height=0.6)
ax.set_yticks(range(len(tasks)), [t[0] for t in tasks[::-1]], fontsize=8)
ax.set_xticks(range(0, 16)); ax.set_xlabel("Month from approval"); ax.grid(axis="y", alpha=0)
for m, lab in [(4, "Gate 1: pilot KPIs"), (9, "Gate 2: 2-hub KPIs"), (12, "Gate 3: full network")]:
    ax.axvline(m, color=C["accent"], ls="--", lw=0.9); ax.text(m + 0.05, len(tasks) - 0.4, lab, fontsize=7, color=C["accent"])
ax.set_title("Fig 8. Implementation roadmap (15 months, stage-gated)")
fig.savefig(FIG / "fig08_gantt.png"); plt.close(fig)
print("figures done")
