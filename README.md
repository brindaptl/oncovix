# Oncovix: Tumor Mutational Burden and Survival After Immunotherapy

Survival estimates after immune checkpoint inhibitor therapy, based on tumor mutational burden (TMB) percentile within cancer type.

Live app: https://oncovix.streamlit.app

## What it does
Enter a patient's cancer type, immunotherapy type, TMB, sex and age group. Oncovix returns an estimated survival curve and survival probabilities at 12 and 24 months. TMB is converted to a percentile within the chosen cancer type, so the same number can mean different things in different cancers.

## How it works
The model is a Cox proportional hazards model stratified by cancer type, fit on the public MSK-IMPACT immunotherapy cohort (Samstein et al., 2019; n = 1,660). It was validated with repeated 5-fold cross-validation and calibration analysis. The app reads saved coefficients and does not retrain anything.

## Files
- app.py: Streamlit app
- model.py, oncovix_core.py: model code
- model_params.json: fitted parameters
- requirements.txt: dependencies
- study/: analysis code and figures for the accompanying manuscript

## Limitations
Validated only internally on one institution's data. No untreated comparison group. Research and educational use only. Not for clinical decisions.
