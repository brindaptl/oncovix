import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

base = os.path.expanduser("~/Desktop/oncovix-study")
df = pd.read_csv(os.path.join(base, "cleaned_samstein.csv"))

age_map = {"<30": 0, "31-50": 1, "50-60": 2, "61-70": 3, ">71": 4}
df["age_ord"] = df["age_group"].map(age_map)
df["log_tmb"] = np.log1p(df["tmb"])
df["tmb_pct"] = df.groupby("cancer_type")["tmb"].rank(pct=True)
df["male"] = (df["sex"] == "Male").astype(int)
drug = pd.get_dummies(df["drug_type"], prefix="drug", drop_first=True, dtype=int)
cancer = pd.get_dummies(df["cancer_type"], prefix="cancer", drop_first=True, dtype=int)

data = pd.concat([df[["os_months", "event", "log_tmb", "tmb_pct", "male", "age_ord"]], drug, cancer], axis=1)
clin = ["male", "age_ord"] + list(drug.columns) + list(cancer.columns)

feature_sets = {
    "A. Clinical only": clin,
    "B. Raw TMB only": ["log_tmb"],
    "C. TMB percentile within cancer type only": ["tmb_pct"],
    "D. Clinical + raw TMB": clin + ["log_tmb"],
    "E. Clinical + TMB percentile": clin + ["tmb_pct"],
}

rows = []
for rep in range(5):
    kf = KFold(n_splits=5, shuffle=True, random_state=rep)
    for fold, (tr, te) in enumerate(kf.split(data)):
        train, test = data.iloc[tr], data.iloc[te]
        for name, feats in feature_sets.items():
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(train[feats + ["os_months", "event"]], "os_months", "event")
            risk = cph.predict_partial_hazard(test[feats])
            c = concordance_index(test["os_months"], -risk, test["event"])
            rows.append({"rep": rep, "fold": fold, "model": name, "c_index": c})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(base, "cv_results.csv"), index=False)

print("\nCross-validated C-index (mean, SD across 25 folds):")
print(res.groupby("model")["c_index"].agg(["mean", "std"]).round(3))

wide = res.pivot_table(index=["rep", "fold"], columns="model", values="c_index")
pairs = [
    ("D. Clinical + raw TMB", "A. Clinical only"),
    ("E. Clinical + TMB percentile", "A. Clinical only"),
    ("C. TMB percentile within cancer type only", "B. Raw TMB only"),
    ("E. Clinical + TMB percentile", "D. Clinical + raw TMB"),
]
print("\nPaired C-index differences (first minus second), mean and middle 95% of fold differences")
print("(descriptive only: CV folds overlap, so this is not a formal confidence interval):")
for a, b in pairs:
    d = wide[a] - wide[b]
    print(f"{a}  vs  {b}: {d.mean():+.4f}  [{d.quantile(0.025):+.4f}, {d.quantile(0.975):+.4f}]")

# Hazard ratios, full-data models D and E (no penalty)
for label, key in [("D", "D. Clinical + raw TMB"), ("E", "E. Clinical + TMB percentile")]:
    cph = CoxPHFitter()
    fd = data[feature_sets[key] + ["os_months", "event"]]
    cph.fit(fd, "os_months", "event")
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4)
    hr.to_csv(os.path.join(base, f"hazard_ratios_model{label}.csv"))
    print(f"\nHazard ratios, full-data model {label}:")
    print(hr)

# Figure 2
order = list(feature_sets.keys())
fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot([res[res["model"] == m]["c_index"] for m in order], tick_labels=[m.split(".")[0] for m in order])
ax.set_ylabel("Cross-validated C-index")
ax.set_xlabel("Model")
ax.set_title("Predictive performance for overall survival (5x5-fold CV)")
plt.tight_layout()
os.makedirs(os.path.join(base, "figures"), exist_ok=True)
plt.savefig(os.path.join(base, "figures", "fig2_cindex.png"), dpi=300)
print("\nSaved figures/fig2_cindex.png")