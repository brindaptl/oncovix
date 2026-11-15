import os
import pandas as pd
from lifelines import CoxPHFitter

pd.set_option("display.width", 200)

base = os.path.expanduser("~/Desktop/oncovix-study")
df = pd.read_csv(os.path.join(base, "cleaned_samstein.csv"))

age_map = {"<30": 0, "31-50": 1, "50-60": 2, "61-70": 3, ">71": 4}
df["age_ord"] = df["age_group"].map(age_map)
df["male"] = (df["sex"] == "Male").astype(int)
df["tmb_pct"] = df.groupby("cancer_type")["tmb"].rank(pct=True)
df["tmb_pct_panel"] = df.groupby(["cancer_type", "Gene Panel"])["tmb"].rank(pct=True)

drug = pd.get_dummies(df["drug_type"], prefix="drug", drop_first=True, dtype=int)
panel = pd.get_dummies(df["Gene Panel"], prefix="panel", drop_first=True, dtype=int)
d = pd.concat([df[["os_months", "event", "tmb_pct", "tmb_pct_panel", "male", "age_ord", "cancer_type"]], drug, panel], axis=1)

base_cov = ["male", "age_ord"] + list(drug.columns)

def fit(label, tmb_col, extra):
    cols = [tmb_col] + base_cov + extra
    m = CoxPHFitter()
    m.fit(d[cols + ["os_months", "event", "cancer_type"]], "os_months", "event", strata=["cancer_type"])
    s = m.summary.loc[tmb_col]
    print(f"{label}: HR {s['exp(coef)']:.3f} (95% CI {s['exp(coef) lower 95%']:.3f} to {s['exp(coef) upper 95%']:.3f}), p = {s['p']:.2g}")

print("Stratified Cox, TMB hazard ratio (highest vs lowest percentile):")
fit("Primary model (as reported)            ", "tmb_pct", [])
fit("+ gene panel as covariate              ", "tmb_pct", list(panel.columns))
fit("Percentile within cancer type x panel  ", "tmb_pct_panel", [])
fit("Percentile within cancer x panel + panel", "tmb_pct_panel", list(panel.columns))