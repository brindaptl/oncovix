import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

base = os.path.expanduser("~/Desktop/oncovix-study")
df = pd.read_csv(os.path.join(base, "cleaned_samstein.csv"))

age_map = {"<30": 0, "31-50": 1, "50-60": 2, "61-70": 3, ">71": 4}
df["age_ord"] = df["age_group"].map(age_map)
df["tmb_pct"] = df.groupby("cancer_type")["tmb"].rank(pct=True)
df["male"] = (df["sex"] == "Male").astype(int)

# ---------- Table 1 ----------
print("=== TABLE 1: patient characteristics ===")
n = len(df)
print(f"Patients: {n}; deaths: {int(df['event'].sum())} ({df['event'].mean():.1%})")
km = KaplanMeierFitter().fit(df["os_months"], 1 - df["event"])
print(f"Median follow-up (reverse Kaplan-Meier): {km.median_survival_time_:.1f} months")
rows = []
for col, label in [("sex", "Sex"), ("age_group", "Age group"), ("drug_type", "ICI type"), ("cancer_type", "Cancer type")]:
    vc = df[col].value_counts()
    for k, v in vc.items():
        rows.append({"variable": label, "level": k, "n": v, "percent": round(100 * v / n, 1)})
t1 = pd.DataFrame(rows)
print(t1.to_string(index=False))
q = df["tmb"].quantile([0.25, 0.5, 0.75])
print(f"TMB overall: median {q[0.5]:.1f} (IQR {q[0.25]:.1f} to {q[0.75]:.1f}) mut/Mb")
t1.to_csv(os.path.join(base, "table1.csv"), index=False)
tmb_by = df.groupby("cancer_type")["tmb"].describe()[["count", "25%", "50%", "75%"]].round(1)
print("\nTMB by cancer type:")
print(tmb_by)

# ---------- Stratified Cox ----------
drug = pd.get_dummies(df["drug_type"], prefix="drug", drop_first=True, dtype=int)
d = pd.concat([df[["os_months", "event", "tmb_pct", "male", "age_ord", "cancer_type"]], drug], axis=1)
cols = ["tmb_pct", "male", "age_ord"] + list(drug.columns)

m0 = CoxPHFitter()
m0.fit(d[cols + ["os_months", "event", "cancer_type"]], "os_months", "event", strata=["cancer_type"])
print("\n=== STRATIFIED COX (separate baseline hazard per cancer type) ===")
print(m0.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4))
m0.summary.to_csv(os.path.join(base, "stratified_cox.csv"))
try:
    ph = proportional_hazard_test(m0, d[cols + ["os_months", "event", "cancer_type"]], time_transform="rank")
    print("\nProportional hazards test, stratified model:")
    print(ph.summary.round(4))
except Exception as e:
    print("\n(PH test on stratified model skipped:", e, ")")

# ---------- Interaction test ----------
cdum = pd.get_dummies(d["cancer_type"], prefix="c", drop_first=True, dtype=int)
inter = cdum.mul(d["tmb_pct"], axis=0).add_prefix("tmbx_")
full = pd.concat([d[cols + ["os_months", "event", "cancer_type"]], inter], axis=1)
m1 = CoxPHFitter()
m1.fit(full, "os_months", "event", strata=["cancer_type"])
lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
dof = inter.shape[1]
print("\n=== INTERACTION TEST: does the TMB effect differ by cancer type? ===")
print(f"Likelihood-ratio chi-square = {lr:.2f}, df = {dof}, p = {chi2.sf(lr, dof):.4f}")

# ---------- Breast cancer re-check ----------
print("\n=== BREAST CANCER CHECK ===")
b = df[df["cancer_type"] == "Breast Cancer"]
print("Sex counts in breast group:")
print(b["sex"].value_counts())
print("Events:", int(b["event"].sum()), "of", len(b))
cov = [c for c in ["tmb_pct", "male", "age_ord"] if b[c].nunique() > 1]
print("Covariates kept (non-constant):", cov)
try:
    mb = CoxPHFitter()
    mb.fit(b[cov + ["os_months", "event"]], "os_months", "event")
    print(mb.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(4))
except Exception as e:
    print("Breast model still failed:", e)