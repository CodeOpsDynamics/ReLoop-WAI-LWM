"""
03 - AI grading engine: predict the value-maximising disposition of each returned corporate IT asset.
Dispositions (value hierarchy of reverse logistics): REDEPLOY > REFURBISH_RESELL > HARVEST_PARTS > RECYCLE
Data: SYNTHETIC asset-return records (disclosed), generated from documented ITAD grading logic + noise,
because enterprise asset registers are confidential. Model: Random Forest (scikit-learn), benchmarked against
(a) the common corporate rule "anything at end of refresh cycle goes to the recycler" and (b) perfect information.
Business metric = value recovered per asset (INR), not just accuracy.
Concept link: LWM Session 6 (reverse logistics: returns gatekeeping, disposition), Session 3-4 (IT & analytics).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from config import SYN, FIG, TAB, C, SEED, save_result

rng = np.random.default_rng(SEED)
N = 12000
TYPES = {"Laptop": (0.45, 2.2), "Desktop": (0.15, 8.0), "Monitor": (0.20, 5.0),
         "Server": (0.05, 25.0), "Network": (0.05, 4.0), "Mobile": (0.10, 0.2)}
CLASSES = ["REDEPLOY", "REFURBISH_RESELL", "HARVEST_PARTS", "RECYCLE"]
# Net value recovered per unit (INR) by asset type and disposition (assumptions; see report Annex C)
VALUE = {"Laptop": [22000, 11000, 2500, 350], "Desktop": [9000, 5000, 1500, 500],
         "Monitor": [4000, 2500, 400, 200], "Server": [60000, 35000, 12000, 1500],
         "Network": [8000, 5000, 1200, 150], "Mobile": [6000, 4000, 600, 80]}
REWORK_COST = 1200   # cost when an asset is routed too optimistically and must be re-handled

import os
FROZEN = SYN / "it_asset_returns_synthetic.csv"
# The synthetic file is generated ONCE and then frozen in the repository: floating-point maths in the random
# generators differs slightly between CPU architectures (e.g. Apple Silicon vs x86), so re-generating on another
# machine can shift a few records. Set RELOOP_REGENERATE=1 to rebuild it from the documented logic below.
if FROZEN.exists() and os.environ.get("RELOOP_REGENERATE") != "1":
    df = pd.read_csv(FROZEN)
    truth = df.best_disposition.map({c: i for i, c in enumerate(CLASSES)}).values
else:
    t = rng.choice(list(TYPES), N, p=[v[0] for v in TYPES.values()])
    age = np.clip(rng.gamma(4.5, 0.95, N), 0.3, 10).round(1)
    brand_tier = rng.choice([1, 2, 3], N, p=[0.55, 0.30, 0.15])          # 1 = enterprise grade
    cosmetic = rng.choice([4, 3, 2, 1], N, p=[0.30, 0.38, 0.22, 0.10])   # 4 = Grade A
    func_fail = (rng.random(N) < 0.06 + 0.035 * age).astype(int)
    storage_health = np.clip(100 - age * rng.uniform(4, 9, N) - func_fail * rng.uniform(10, 40, N), 5, 100).round()
    battery_health = np.where(np.isin(t, ["Laptop", "Mobile"]),
                              np.clip(100 - age * rng.uniform(7, 13, N), 5, 100), np.nan).round()
    cpu_gen_gap = np.clip((age * 0.9 + rng.normal(0, 0.8, N)).round(), 0, 10)   # generations behind current
    wipe_verified = (rng.random(N) > 0.03 + 0.04 * func_fail).astype(int)        # NIST 800-88 wipe passed
    city = rng.choice(["Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai", "Gurugram", "Noida", "Kolkata"], N,
                      p=[0.20, 0.15, 0.13, 0.12, 0.11, 0.13, 0.10, 0.06])

    # Latent "true best" disposition (grading expert logic + noise)
    score = (3.2 - 0.42 * age - 0.25 * (brand_tier - 1) + 0.35 * (cosmetic - 2.5) + 0.018 * (storage_health - 70)
             - 1.9 * func_fail - 0.20 * cpu_gen_gap + rng.normal(0, 0.55, N))
    bh = np.nan_to_num(battery_health, nan=80)
    score -= np.where(bh < 55, 0.6, 0)
    score += np.where(t == "Server", 0.5, 0) + np.where(t == "Monitor", 0.6, 0)
    truth = np.select([score > 1.6, score > 0.3, score > -1.3], [0, 1, 2], 3)
    truth = np.where(wipe_verified == 0, np.maximum(truth, 2), truth)    # un-wipeable media cannot leave as an asset

    df = pd.DataFrame(dict(asset_id=[f"A{i:05d}" for i in range(N)], asset_type=t, city=city, age_yrs=age,
                           brand_tier=brand_tier, cosmetic_grade=cosmetic, functional_fail=func_fail,
                           storage_health_pct=storage_health, battery_health_pct=battery_health,
                           cpu_gen_gap=cpu_gen_gap, wipe_verified=wipe_verified,
                           weight_kg=[TYPES[x][1] for x in t], best_disposition=[CLASSES[i] for i in truth]))
    df.to_csv(SYN / "it_asset_returns_synthetic.csv", index=False)

X = pd.get_dummies(df.drop(columns=["asset_id", "best_disposition", "city", "weight_kg"]), columns=["asset_type"])
X["battery_health_pct"] = X.battery_health_pct.fillna(-1)          # -1 = not applicable (no battery)
yv = truth
Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, yv, df.index, test_size=0.25, stratify=yv, random_state=SEED)
rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3, class_weight="balanced",
                            random_state=SEED, n_jobs=-1)
cv = cross_val_score(rf, Xtr, ytr, cv=5, scoring="f1_macro")
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)

# Rule-based baseline used by many corporates: past refresh age (>=4 yrs) or failed -> recycler, else redeploy
test = df.loc[ite].copy()
rule = np.where((test.age_yrs >= 4) | (test.functional_fail == 1), 3, 0)

def realised(pred_cls, true_cls, types):
    v = []
    for p, tr, ty in zip(pred_cls, true_cls, types):
        if p == tr: v.append(VALUE[ty][tr])
        elif p < tr: v.append(VALUE[ty][tr] - REWORK_COST)     # over-optimistic -> rework then true value
        else: v.append(VALUE[ty][p])                            # too pessimistic -> value left on table
    return np.array(v)

v_rule = realised(rule, yte, test.asset_type)
v_ml = realised(pred, yte, test.asset_type)
v_perf = realised(yte, yte, test.asset_type)
uplift_per_asset = v_ml.mean() - v_rule.mean()
capture = (v_ml.sum() - v_rule.sum()) / (v_perf.sum() - v_rule.sum())

cm = confusion_matrix(yte, pred)
rep = classification_report(yte, pred, target_names=CLASSES, output_dict=True)
pd.DataFrame(rep).T.to_csv(TAB / "t03_classification_report.csv")
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
imp.to_csv(TAB / "t03_feature_importance.csv")

# Fig 3: value comparison + feature importance
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1, 1.2]))
vals = [v_rule.mean(), v_ml.mean(), v_perf.mean()]
b = axes[0].bar(["Rule: age>=4\n-> recycle", "AI grading\n(Random Forest)", "Perfect\ninformation"], vals,
                color=[C["amber"], C["green"], C["light"]])
for r, v in zip(b, vals): axes[0].text(r.get_x() + r.get_width() / 2, v + 150, f"Rs {v:,.0f}", ha="center", fontsize=9)
axes[0].set_ylabel("Avg value recovered per returned asset (INR)")
axes[0].set_title("Fig 3a. Value recovered per asset")
top = imp.head(8)[::-1]
axes[1].barh(top.index.str.replace("_", " "), top.values, color=C["green"])
axes[1].set_title("Fig 3b. What drives the disposition decision")
axes[1].set_xlabel("Random-forest feature importance")
fig.savefig(FIG / "fig03_disposition_ml.png"); plt.close(fig)

# Confusion matrix figure for annex
fig, ax = plt.subplots(figsize=(5, 4.2))
ax.imshow(cm, cmap="Greens"); ax.grid(False)
ax.set_xticks(range(4), [c.replace("_", "\n") for c in CLASSES], fontsize=7); ax.set_yticks(range(4), CLASSES, fontsize=7)
for i in range(4):
    for j in range(4): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set, n=3,000)")
fig.savefig(FIG / "figA_confusion_matrix.png"); plt.close(fig)

mix = df.best_disposition.value_counts(normalize=True).round(3).to_dict()
wt_share_recycle = df.loc[df.best_disposition == "RECYCLE", "weight_kg"].sum() / df.weight_kg.sum()
wt_share_recycle_harvest = df.loc[df.best_disposition.isin(["RECYCLE", "HARVEST_PARTS"]), "weight_kg"].sum() / df.weight_kg.sum()
avg_wt = df.weight_kg.mean()
res = dict(n_records=N, accuracy=round(accuracy_score(yte, pred), 3), f1_macro=round(f1_score(yte, pred, average="macro"), 3),
           cv_f1_mean=round(cv.mean(), 3), cv_f1_sd=round(cv.std(), 3),
           rule_accuracy=round(accuracy_score(yte, rule), 3),
           value_rule=round(v_rule.mean()), value_ml=round(v_ml.mean()), value_perfect=round(v_perf.mean()),
           uplift_per_asset=round(uplift_per_asset), uplift_pct=round(100 * uplift_per_asset / v_rule.mean(), 1),
           gap_captured_pct=round(100 * capture, 1), top_features=imp.head(5).round(3).to_dict(),
           disposition_mix=mix, weight_share_to_recycler=round(wt_share_recycle, 3),
           weight_share_recycle_or_harvest=round(wt_share_recycle_harvest, 3), avg_weight_kg=round(avg_wt, 2),
           rule_share_recycled=round(float((rule == 3).mean()), 3))
save_result("disposition_ml", res); print(res)
print(classification_report(yte, pred, target_names=CLASSES))
