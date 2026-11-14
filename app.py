import streamlit as st
import plotly.graph_objects as go
from oncovix_core import load_params, build_features, survival_curve, survival_at

st.set_page_config(page_title="Oncovix", layout="wide")

@st.cache_data
def get_params():
    return load_params("model_params.json")

P = get_params()

st.title("Oncovix")
st.caption("Research prototype: survival estimates for patients treated with immune checkpoint inhibitors")
st.warning(
    "Research prototype only. This tool is not validated for clinical use and must not guide any "
    "medical decision. Estimates come from a Cox model fit to a public cohort and show modest "
    "discrimination (cross-validated C-index about 0.63)."
)

with st.sidebar:
    st.header("Patient inputs")
    cancer = st.selectbox("Cancer type", P["cancer_types"])
    drug = st.selectbox("Immunotherapy type", P["drug_types"])
    tmb = st.number_input("Tumor mutational burden (mutations/Mb)", 0.0, 250.0, 6.0, 0.5)
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.selectbox("Age group at diagnosis (years)", list(P["age_map"].keys()), index=3)

feats = build_features(P, cancer, drug, tmb, sex, age)
ts, s = survival_curve(P, feats, tmax=60)
pct = feats["tmb_pct"]

c1, c2, c3 = st.columns(3)
c1.metric("TMB percentile within cancer type", f"{pct*100:.0f}%")
c2.metric("Estimated 12-month survival", f"{survival_at(P, feats, 12)*100:.0f}%")
c3.metric("Estimated 24-month survival", f"{survival_at(P, feats, 24)*100:.0f}%")

fig = go.Figure(go.Scatter(x=ts, y=s, mode="lines", line=dict(shape="hv", width=3)))
fig.update_layout(template="plotly_dark", xaxis_title="Months since immunotherapy start",
                  yaxis_title="Estimated survival probability", yaxis=dict(range=[0, 1.02]),
                  height=420)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"**About the model.** Cox proportional hazards model (ridge penalty {P['penalizer']}) trained on "
    f"{P['n_patients']:,} patients ({P['n_events']:,} deaths) from the MSK-IMPACT immunotherapy cohort "
    "(Samstein et al., 2019; cBioPortal). Predictors: TMB percentile within cancer type, cancer type, "
    "immunotherapy type, sex and age group. Out-of-fold calibration was acceptable at 12 and 24 months, "
    "but the model has not been externally validated, and the curve is a point estimate without an "
    "uncertainty interval. Results reflect one institution's patients and do not describe treatment effect."
)
