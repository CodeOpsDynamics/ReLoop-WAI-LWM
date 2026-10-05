"""
02 - Forecasting the FORMAL reverse-logistics flow (tonnes that must physically move to authorised recyclers).
Why this target: the logistics network is sized on what moves through formal channels, not on total generation.
Approach = triangulation of two models (a single ML model on 7 annual points would be over-fitting):
  M1  Logistic (S-curve) diffusion fitted to the formal-processing series FY18-FY25 (scipy curve_fit)
  M2  Policy-driven model: generation (FY25 base, 3 growth scenarios) x EPR recycling-target trajectory
      (E-Waste Rules 2022: 60% FY24-25, 70% FY26-27, 80% FY28 onward)
Concept link: LWM Session 1 (demand uncertainty), Session 6 (reverse-logistics volume planning).
NOTE: Prophet/ARIMA were tested in the prompt log and rejected: 7 points + a 2023 methodology break.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from config import RAW, FIG, TAB, C, save_result

nat = pd.read_csv(RAW / "national_series.csv")
x = nat.fy_end_year.values.astype(float)
y = nat.formally_processed_t.values / 1e5          # lakh tonnes

def logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

popt, pcov = curve_fit(logistic, x, y, p0=[15, 0.5, 2024], bounds=([8, 0.05, 2015], [40, 2.0, 2035]), maxfev=20000)
resid = y - logistic(x, *popt)
mape = np.mean(np.abs(resid / y)) * 100
rng = np.random.default_rng(42)
yrs = np.arange(2018, 2031)
# Parameter-uncertainty band via Monte-Carlo draws from the fitted covariance
draws = rng.multivariate_normal(popt, pcov, 4000)
draws = draws[(draws[:, 0] > 8) & (draws[:, 1] > 0)]
sims = np.array([logistic(yrs, *d) for d in draws])
lo, hi = np.percentile(sims, [10, 90], axis=0)
m1 = logistic(yrs, *popt)

# M2: policy-driven scenarios
gen25 = nat.loc[nat.fy == "2024-25", "generated_t"].item() / 1e5
target = {2026: 0.70, 2027: 0.70, 2028: 0.80, 2029: 0.80, 2030: 0.80}
achieve = {2026: 0.73, 2027: 0.76, 2028: 0.80, 2029: 0.83, 2030: 0.85}   # compliance trend (FY24 61.9% -> FY25 70.7%)
scen = {"Low (8% gen. growth)": 0.08, "Base (11.5%)": 0.115, "High (14%)": 0.14}
rows = []
for name, g in scen.items():
    for yr in range(2026, 2031):
        gen = gen25 * (1 + g) ** (yr - 2025)
        rows.append(dict(scenario=name, fy_end_year=yr, generation_lakh_t=gen,
                         formal_flow_lakh_t=gen * achieve[yr], epr_floor_lakh_t=gen * target[yr]))
m2 = pd.DataFrame(rows)
m2.to_csv(TAB / "t02_policy_forecast.csv", index=False)
fc = pd.DataFrame(dict(fy_end_year=yrs, logistic_lakh_t=m1, p10=lo, p90=hi))
fc.to_csv(TAB / "t02_logistic_forecast.csv", index=False)

base30 = m2[(m2.scenario.str.startswith("Base")) & (m2.fy_end_year == 2030)].formal_flow_lakh_t.item()
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.fill_between(yrs, lo, hi, color=C["light"], alpha=0.7, label="M1 logistic: P10-P90 band")
ax.plot(yrs, m1, color=C["green"], lw=2, label=f"M1 logistic fit (MAPE {mape:.1f}%)")
for (name, g), ls in zip(scen.items(), [":", "--", "-."]):
    s = m2[m2.scenario == name]
    ax.plot(s.fy_end_year, s.formal_flow_lakh_t, ls, color=C["accent"], lw=1.6, label=f"M2 policy: {name}")
ax.scatter(x, y, color=C["dark"], zorder=5, label="Actual formal processing (CPCB)")
ax.axvline(2023, color=C["grey"], lw=0.8, ls="--"); ax.text(2023.1, 1, "E-Waste Rules 2022\nin force", fontsize=8, color=C["grey"])
cap = pd.read_csv(TAB / "t01_recycler_registry_clean.csv").capacity_tpa.sum() / 1e5
ax.axhline(cap, color=C["grey"], lw=1.2, ls="-."); ax.text(2022.2, cap + 0.8, f"Authorised capacity, Aug 2026 (CPCB registry): {cap:.1f} lakh t", fontsize=8, color=C["grey"])
ax.set_ylim(0, cap * 1.12)
ax.set_ylabel("Lakh tonnes per year"); ax.set_xlabel("Financial year ending")
ax.set_title("Fig 2. Formal e-waste reverse-logistics flow: actuals and FY26-FY30 forecast")
ax.legend(fontsize=7.5, frameon=False, loc="center left")
fig.savefig(FIG / "fig02_forecast.png"); plt.close(fig)

res = dict(K_saturation_lakh_t=round(popt[0], 2), growth_r=round(popt[1], 3), inflection_year=round(popt[2], 1),
           mape_pct=round(mape, 1), m1_2030=round(m1[-1], 2), m1_2030_p10=round(lo[-1], 2), m1_2030_p90=round(hi[-1], 2),
           m2_2030_low=round(m2[(m2.scenario.str.startswith("Low")) & (m2.fy_end_year == 2030)].formal_flow_lakh_t.item(), 2),
           m2_2030_base=round(base30, 2),
           m2_2030_high=round(m2[(m2.scenario.str.startswith("High")) & (m2.fy_end_year == 2030)].formal_flow_lakh_t.item(), 2),
           fy25_actual_lakh_t=round(y[-1], 2), growth_multiple_base=round(base30 / y[-1], 2),
           util_2030_high_pct=round(100 * m2[(m2.scenario.str.startswith("High")) & (m2.fy_end_year == 2030)].formal_flow_lakh_t.item() / cap, 1))
save_result("forecast", res); print(res)
