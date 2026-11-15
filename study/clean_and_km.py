import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

base = os.path.expanduser("~/Desktop/oncovix-study")
raw = os.path.join(base, "samstein_pancancer", "tmb_mskcc_2018_clinical_data.tsv")
os.makedirs(os.path.join(base, "figures"), exist_ok=True)

df = pd.read_csv(raw, sep="\t")

# Rename columns for easier use
df = df.rename(columns={
    "Patient ID": "patient_id",
    "Cancer Type": "cancer_type",
    "Drug Type": "drug_type",
    "TMB (nonsynonymous)": "tmb",
    "Overall Survival (Months)": "os_months",
    "Overall Survival Status": "os_status",
    "Sex": "sex",
    "Age Group at Diagnosis in Years": "age_group",
})

# Event: 1 = deceased, 0 = living (censored)
df["event"] = df["os_status"].str.startswith("1").astype(int)

# Checks
print("Duplicate patient IDs:", df["patient_id"].duplicated().sum())
print()
print("Age group values:")
print(df["age_group"].value_counts(dropna=False))
print()
print("Sex values:")
print(df["sex"].value_counts(dropna=False))
print()

# Drop cancer types with fewer than 10 patients
counts = df["cancer_type"].value_counts()
keep = counts[counts >= 10].index
dropped = counts[counts < 10]
print("Dropped cancer types (n < 10):")
print(dropped)
df = df[df["cancer_type"].isin(keep)].copy()

# TMB high = top 20% within each cancer type
df["tmb_high"] = df.groupby("cancer_type")["tmb"].transform(lambda s: s >= s.quantile(0.8)).astype(int)

print()
print("Patients after cleaning:", len(df))
print("TMB-high patients:", df["tmb_high"].sum())

# Save cleaned data
out = os.path.join(base, "cleaned_samstein.csv")
df.to_csv(out, index=False)
print("Saved:", out)

# Kaplan-Meier: TMB-high vs rest
high = df[df["tmb_high"] == 1]
low = df[df["tmb_high"] == 0]

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7, 5))
kmf.fit(high["os_months"], high["event"], label="TMB-high (top 20% within cancer type)")
kmf.plot_survival_function(ax=ax, ci_show=True)
kmf.fit(low["os_months"], low["event"], label="TMB-low (remaining 80%)")
kmf.plot_survival_function(ax=ax, ci_show=True)

res = logrank_test(high["os_months"], low["os_months"], high["event"], low["event"])
print()
print("Median OS TMB-high:", KaplanMeierFitter().fit(high["os_months"], high["event"]).median_survival_time_)
print("Median OS TMB-low:", KaplanMeierFitter().fit(low["os_months"], low["event"]).median_survival_time_)
print("Log-rank p-value:", res.p_value)

ax.set_xlabel("Months")
ax.set_ylabel("Overall survival probability")
ax.set_title("Overall survival by TMB group")
plt.tight_layout()
plt.savefig(os.path.join(base, "figures", "fig1_km_tmb.png"), dpi=300)
print("Saved figure: figures/fig1_km_tmb.png")