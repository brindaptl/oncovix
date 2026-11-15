import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

base = os.path.expanduser("~/Desktop/oncovix-study")
df = pd.read_csv(os.path.join(base, "cleaned_samstein.csv"))

age_map = {"<30": 0, "31-50": 1, "50-60": 2, "61-70": 3, ">71": 4}
df["age_ord"] = df["age_group"].map(age_map)
df["tmb_pct"] = df.groupby("cancer_type")["tmb"].rank(pct=True)
df["male"] = (df["sex"] == "Male").astype(int)
drug = pd.get_dummies(df["drug_type"], prefix="drug", drop_first=True, dtype=int)
cancer = pd.get_dummies(df["cancer_type"], prefix="cancer", drop_first=True, dtype=int)
data = pd.concat([df[["os_months", "event", "tmb_pct", "male", "age_ord"]], drug, cancer], axis=1)

clin = ["male", "age_ord"] + list(drug.columns) + list(cancer.columns)
featA = clin
featE = clin + ["tmb_pct"]

# 1. Proportional hazards check (model E)
cph = CoxPHFitter()
fd = data[featE + ["os_months", "event"]]
cph.fit(fd, "os_months", "event")
ph = proportional_hazard_test(cph, fd, time_transform="rank")
print("\n1. Proportional hazards test (model E). Small p suggests the assumption may be violated:")
print(ph.summary.round(4))

# 2. Bootstrap CI for C-index gain (E minus A), evaluated on out-of-bag patients
rng = np.random.default_rng(42)
n = len(data)
deltas = []
for b in range(200):
    idx = rng.integers(0, n, n)
    train = data.iloc[idx]
    oob = np.setdiff1d(np.arange(n), np.unique(idx))
    test = data.iloc[oob]
    cs = {}
    for name, feats in [("A", featA), ("E", featE)]:
        m = CoxPHFitter(penalizer=0.01)
        m.fit(train[feats + ["os_months", "event"]], "os_months", "event")
        risk = m.predict_partial_hazard(test[feats])
        cs[name] = concordance_index(test["os_months"], -risk, test["event"])
    deltas.append(cs["E"] - cs["A"])
deltas = np.array(deltas)
print("\n2. Bootstrap (200 resamples, out-of-bag evaluation): C-index gain from adding TMB percentile to clinical features")
print(f"   mean {deltas.mean():+.4f}, 95% interval [{np.percentile(deltas, 2.5):+.4f}, {np.percentile(deltas, 97.5):+.4f}]")

# 3. Per-cancer-type TMB effect (exploratory)
rows = []
for ct, g in df.groupby("cancer_type"):
    keep = [c for c in ["tmb_pct", "male", "age_ord"] if g[c].nunique() > 1]
    g = g[keep + ["os_months", "event"]]
    try:
        m = CoxPHFitter()
        m.fit(g, "os_months", "event")
        s = m.summary.loc["tmb_pct"]
        rows.append({"cancer_type": ct, "n": len(g), "events": int(g["event"].sum()),
                     "HR": s["exp(coef)"], "lo": s["exp(coef) lower 95%"],
                     "hi": s["exp(coef) upper 95%"], "p": s["p"]})
    except Exception as e:
        print("Skipped", ct, "->", e)
res = pd.DataFrame(rows).sort_values("HR")
res.to_csv(os.path.join(base, "per_cancer_tmb_hr.csv"), index=False)
print("\n3. TMB percentile hazard ratio within each cancer type (adjusted for sex and age group where they vary):")
print(res.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
y = np.arange(len(res))
ax.errorbar(res["HR"], y, xerr=[res["HR"] - res["lo"], res["hi"] - res["HR"]], fmt="o", capsize=3)
ax.axvline(1, linestyle="--", color="gray")
ax.set_yticks(y)
ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(res["cancer_type"], res["n"])])
ax.set_xscale("log")
ax.set_xlabel("Hazard ratio for TMB percentile (log scale; below 1 = better survival)")
ax.set_title("TMB and survival within each cancer type")
plt.tight_layout()
os.makedirs(os.path.join(base, "figures"), exist_ok=True)
plt.savefig(os.path.join(base, "figures", "fig3_forest.png"), dpi=300)
print("\nSaved figures/fig3_forest.png")