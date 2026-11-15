import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts

base = os.path.expanduser("~/Desktop/oncovix-study")
figdir = os.path.join(base, "figures")
os.makedirs(figdir, exist_ok=True)
df = pd.read_csv(os.path.join(base, "cleaned_samstein.csv"))

# ---------- Figure 1: Kaplan-Meier with number at risk ----------
high = df[df["tmb_high"] == 1]
low = df[df["tmb_high"] == 0]
k1 = KaplanMeierFitter().fit(high["os_months"], high["event"], label="TMB-high (top 20% within cancer type)")
k2 = KaplanMeierFitter().fit(low["os_months"], low["event"], label="TMB-low (remaining 80%)")
fig, ax = plt.subplots(figsize=(7.5, 6.2))
ax.set_xticks(range(0, 81, 10))
k1.plot_survival_function(ax=ax, ci_show=True)
k2.plot_survival_function(ax=ax, ci_show=True)
add_at_risk_counts(k1, k2, ax=ax)
ax.set_xlim(0, 80)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Months since start of follow-up")
ax.set_ylabel("Overall survival probability")
ax.set_title("Overall survival by TMB group")
plt.tight_layout()
plt.savefig(os.path.join(figdir, "fig1_km_tmb.png"), dpi=300)
plt.close()

# ---------- Figure 2: C-index, named models + paired difference ----------
cv = pd.read_csv(os.path.join(base, "cv_results.csv"))
order = ["A. Clinical only", "B. Raw TMB only", "C. TMB percentile within cancer type only",
         "D. Clinical + raw TMB", "E. Clinical + TMB percentile"]
summ = cv.groupby("model")["c_index"].agg(["mean", "std"]).loc[order]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
y = np.arange(len(order))[::-1]
a1.errorbar(summ["mean"], y, xerr=summ["std"], fmt="o", capsize=3)
a1.axvline(0.5, linestyle="--", color="gray")
a1.set_yticks(y)
a1.set_yticklabels(order)
a1.set_xlim(0.48, 0.68)
a1.set_xlabel("Cross-validated C-index (mean ± SD over 25 folds)")
a1.set_title("a. Discrimination by model")
wide = cv.pivot_table(index=["rep", "fold"], columns="model", values="c_index")
dlt = wide["E. Clinical + TMB percentile"] - wide["A. Clinical only"]
a2.hist(dlt, bins=10)
a2.axvline(0, linestyle="--", color="gray")
a2.axvline(dlt.mean(), color="black")
a2.set_xlabel("Change in C-index, E minus A (per fold)")
a2.set_ylabel("Number of folds")
a2.set_title("b. Gain from adding TMB percentile")
plt.tight_layout()
fig.text(0.005, 0.995, "2)", fontsize=16, fontweight="bold", va="top")
plt.savefig(os.path.join(figdir, "fig2_cindex.png"), dpi=300)
plt.close()

# ---------- Figure 3: forest plot incl. breast and pooled estimate ----------
age_map = {"<30": 0, "31-50": 1, "50-60": 2, "61-70": 3, ">71": 4}
df["age_ord"] = df["age_group"].map(age_map)
df["tmb_pct"] = df.groupby("cancer_type")["tmb"].rank(pct=True)
df["male"] = (df["sex"] == "Male").astype(int)

rows = []
for ct, g in df.groupby("cancer_type"):
    keep = [c for c in ["tmb_pct", "male", "age_ord"] if g[c].nunique() > 1]
    m = CoxPHFitter()
    m.fit(g[keep + ["os_months", "event"]], "os_months", "event")
    s = m.summary.loc["tmb_pct"]
    rows.append({"label": f"{ct} (n={len(g)})", "HR": s["exp(coef)"],
                 "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"]})
per = pd.DataFrame(rows).sort_values("HR", ascending=False).reset_index(drop=True)

drug = pd.get_dummies(df["drug_type"], prefix="drug", drop_first=True, dtype=int)
d2 = pd.concat([df[["os_months", "event", "tmb_pct", "male", "age_ord", "cancer_type"]], drug], axis=1)
ms = CoxPHFitter()
ms.fit(d2, "os_months", "event", strata=["cancer_type"])
s = ms.summary.loc["tmb_pct"]
pooled = {"label": f"All patients, stratified (n={len(df)})", "HR": s["exp(coef)"],
          "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"]}

fig, ax = plt.subplots(figsize=(9, 5.8))
ypos = np.arange(len(per))
ax.errorbar(per["HR"], ypos, xerr=[per["HR"] - per["lo"], per["hi"] - per["HR"]], fmt="o", capsize=3, color="tab:blue")
yp = len(per) + 0.8
ax.errorbar([pooled["HR"]], [yp], xerr=[[pooled["HR"] - pooled["lo"]], [pooled["hi"] - pooled["HR"]]],
            fmt="D", capsize=3, color="black")
ax.axvline(1, linestyle="--", color="gray")
ax.set_yticks(list(ypos) + [yp])
ax.set_yticklabels(list(per["label"]) + [pooled["label"]])
ax.set_xscale("log")
ticks = [0.1, 0.2, 0.5, 1, 2, 5]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlim(0.1, 5)
ax.set_xlabel("Hazard ratio for TMB percentile, highest vs lowest (log scale; below 1 = longer survival)")
ax.set_title("Association of TMB with survival, by cancer type")
for yy, r in list(zip(ypos, per.to_dict("records"))) + [(yp, pooled)]:
    ax.text(1.02, yy, f"{r['HR']:.2f} ({r['lo']:.2f} to {r['hi']:.2f})", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
plt.tight_layout()
fig.text(0.005, 0.995, "3)", fontsize=16, fontweight="bold", va="top")
plt.savefig(os.path.join(figdir, "fig3_forest.png"), dpi=300, bbox_inches="tight")
plt.close()

print("Saved fig1_km_tmb.png, fig2_cindex.png, fig3_forest.png in", figdir)
print("Forest plot rows:", len(per), "cancer types + pooled estimate")