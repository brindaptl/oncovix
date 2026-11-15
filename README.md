#Oncovix

Survival estimates after immune checkpoint inhibitor therapy, based on tumor mutational burden (TMB) percentile within cancer type.

Live app: https://oncovix.streamlit.app

The model is a Cox proportional hazards model stratified by cancer type, fit on the public MSK-IMPACT immunotherapy cohort (Samstein et al., 2019; n = 1,660). It was validated with repeated 5-fold cross-validation and calibration analysis.

Files: app.py (Streamlit app), model.py and oncovix_core.py (model code), model_params.json (fitted parameters), requirements.txt.

Analysis code and figures for the accompanying manuscript are in the study/ folder.

Research and educational use only. Not for clinical decisions.
