import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from lifelines.plotting import add_at_risk_counts

base = os.path.expanduser("~/Desktop/oncovix-study")
figdir = os.path.join(base, "figures")
os.makedirs(figdir, exist_ok=True)
df = pd.read_csv(os.path.join(base, "cleaned_samstein.csv"))
high = df[df["tmb_high"] == 1]
low = df[df["tmb_high"] == 0]

k1 = KaplanMeierFitter().fit(high["os_months"], high["event"], label=f"TMB-high (n = {len(high)})")
k2 = KaplanMeierFitter().fit(low["os_months"], low["event"], label=f"TMB-low (n = {len(low)})")

def med(k):
    m = k.median_survival_time_
    ci = median_survival_times(k.confidence_interval_)
    return m, ci.iloc[0, 0], ci.iloc[0, 1]

m1, l1, u1 = med(k1)
m2, l2, u2 = med(k2)
p = logrank_test(high["os_months"], low["os_months"], high["event"], low["event"]).p_value
p_text = "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"

age_map = {"<30": 0, "31-50": 1, "50-60": 2, "61-70": 3, ">71": 4}
df["age_ord"] = df["age_group"].map(age_map)
df["tmb_pct"] = df.groupby("cancer_type")["tmb"].rank(pct=True)
df["male"] = (df["sex"] == "Male").astype(int)
drug = pd.get_dummies(df["drug_type"], prefix="drug", drop_first=True, dtype=int)
d2 = pd.concat([df[["os_months", "event", "tmb_pct", "male", "age_ord", "cancer_type"]], drug], axis=1)
ms = CoxPHFitter().fit(d2, "os_months", "event", strata=["cancer_type"])
s = ms.summary.loc["tmb_pct"]
hr = f"{s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f})"

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.set_xticks(range(0, 81, 10))
k1.plot_survival_function(ax=ax, ci_show=True, color="tab:blue")
k2.plot_survival_function(ax=ax, ci_show=True, color="tab:orange")
for m, col in [(m1, "tab:blue"), (m2, "tab:orange")]:
    ax.plot([0, m], [0.5, 0.5], color=col, linestyle="--", linewidth=1)
    ax.plot([m, m], [0, 0.5], color=col, linestyle="--", linewidth=1)
    ax.text(m + 1, 0.02, f"median {m:.1f} mo", color=col, fontsize=9, fontweight="bold")
add_at_risk_counts(k1, k2, ax=ax)
ax.set_xlim(0, 80)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Months since start of follow-up")
ax.set_ylabel("Overall survival probability")
ax.set_title("Overall survival by TMB group", fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.text(-0.12, 1.06, "1)", transform=ax.transAxes, fontsize=16, fontweight="bold")

box = (
    "Median OS (95% CI)\n"
    f"High: {m1:.1f} mo ({l1:.1f}-{u1:.1f})\n"
    f"Low: {m2:.1f} mo ({l2:.1f}-{u2:.1f})\n"
    f"Log-rank {p_text}\n"
    f"Adjusted HR {hr}"
)
ax.text(0.57, 0.58, box, transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgray"))

plt.tight_layout()
out = os.path.join(figdir, "fig1_km_final.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print("Saved", out)
