"""Oncovix core: survival estimates from the exported Cox model (model_params.json).
No lifelines needed at runtime. Mirrors final_model.py (validated: Check 1 diff 0.0, Check 2 diff ~1e-16)."""
import json
import numpy as np


def load_params(path="model_params.json"):
    with open(path) as f:
        return json.load(f)


def tmb_percentile(params, cancer_type, x):
    arr = np.array(params["tmb_reference"][cancer_type])
    less = (arr < x).sum()
    eq = (arr == x).sum()
    return float(min((less + (eq + 1) / 2) / len(arr), 1.0))


def build_features(params, cancer_type, drug_type, tmb, sex, age_group):
    ref = params["reference_levels"]
    feats = {f: 0.0 for f in params["features"]}
    feats["male"] = 1.0 if sex == "Male" else 0.0
    feats["age_ord"] = float(params["age_map"][age_group])
    feats["tmb_pct"] = tmb_percentile(params, cancer_type, tmb)
    if drug_type != ref["drug_type"]:
        feats[f"drug_{drug_type}"] = 1.0
    if cancer_type != ref["cancer_type"]:
        feats[f"cancer_{cancer_type}"] = 1.0
    return feats


def survival_at(params, feats, t):
    lp = sum((feats[f] - params["mean"][f]) * params["coef"][f] for f in params["features"])
    bt = np.array(params["baseline_times"])
    bs = np.array(params["baseline_survival"])
    i = np.searchsorted(bt, t, side="right") - 1
    base = 1.0 if i < 0 else bs[i]
    return float(base ** np.exp(lp))


def survival_curve(params, feats, tmax=60, step=1):
    ts = np.arange(0, tmax + step, step)
    return ts, np.array([survival_at(params, feats, t) for t in ts])
