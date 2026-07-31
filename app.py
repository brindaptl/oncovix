# ============================================
# Oncovix - AI Cancer Immunotherapy Predictor
# Created by Brinda Patel, 2026
# All rights reserved
# Creative Commons Attribution-NonCommercial 4.0
# ============================================

import streamlit as st
import plotly.express as px
import pandas as pd
from model import predict_therapy, ALL_CANCER_TYPES, ALL_GENES, DRUG_INFO

st.set_page_config(
    page_title="Oncovix",
    layout="wide"
)

st.markdown("""
    <head>
        <meta property="og:title" content="Oncovix — AI Cancer Immunotherapy Predictor"/>
        <meta property="og:description" content="AI-powered tool that analyzes tumor mutation profiles across 30 cancer types to recommend personalized immunotherapy treatments."/>
        <meta property="og:image" content="https://raw.githubusercontent.com/brindaptl/oncovix/main/preview.png"/>
    </head>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
    }
    .main-title {
        font-size: 5rem;
        font-weight: 900;
        color: #ffffff;
        text-align: center;
        letter-spacing: 8px;
        text-transform: uppercase;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .subtitle {
        font-size: 0.85rem;
        color: #888888;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }
    .built-by {
        font-size: 0.75rem;
        color: #555555;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .disclaimer-box {
        border-left: 3px solid #a8c0d6;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #888888;
        margin-bottom: 16px;
        background-color: #1a1a1a;
    }
    .drug-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 8px;
        border-left: 3px solid #a8a8a8;
        margin-bottom: 12px;
        color: #ffffff;
        font-size: 0.8rem;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        font-size: 0.75rem;
        letter-spacing: 1px;
        color: #555555;
        border-top: 1px solid #222222;
        margin-top: 3rem;
    }
    div[data-testid="stTab"] button {
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        font-size: 0.75rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='padding: 3rem 0 1.5rem 0;'>
        <p style='font-size:10rem; font-weight:900; color:#ffffff; text-align:center; letter-spacing:8px; text-transform:uppercase; margin:0; padding:0;'>Oncovix</p>
        <p style='font-size:1rem; color:#888888; text-align:center; letter-spacing:3px; text-transform:uppercase; margin-top:0.5rem;'>AI-Powered Personalized Cancer Immunotherapy Predictor</p>
        <p style='font-size:0.8rem; color:#555555; text-align:center; letter-spacing:3px; text-transform:uppercase; margin-top:0.3rem;'>Brinda Patel</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='disclaimer-box'>
        <b>Disclaimer:</b> This tool is for research and educational purposes only.
        It is not intended for clinical use and should never replace professional
        medical advice. Always consult a qualified oncologist.
    </div>
""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["  Predict Treatment  ", "  About the Tool  ", "  About the Researcher  "])

with tab1:
    st.header("Patient Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex_option = st.selectbox("Sex", ["Female", "Male", "Other (specify below)"])
        if sex_option == "Other (specify below)":
            sex = st.text_input("Specify Sex", placeholder="e.g. Non-binary, Intersex")
        else:
            sex = sex_option
        unit = st.radio("Unit System", ["Imperial (lbs/ft)", "Metric (kg/cm)"])

    with col2:
        if unit == "Imperial (lbs/ft)":
            height_ft = st.number_input("Height (ft)", min_value=1, max_value=8, value=5)
            height_in = st.number_input("Height (in)", min_value=0, max_value=11, value=5)
            weight_lbs = st.number_input("Weight (lbs)", min_value=50, max_value=700, value=150)
            height_m = ((height_ft * 12) + height_in) * 0.0254
            weight_kg = weight_lbs * 0.453592
        else:
            height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
            weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=300, value=70)
            height_m = height_cm / 100
        bmi = round(weight_kg / (height_m ** 2), 1)
        st.metric("BMI (Auto Calculated)", bmi)

    with col3:
        ecog = st.selectbox("ECOG Performance Status", [
            "0 — Fully active",
            "1 — Restricted but ambulatory",
            "2 — Ambulatory, self-care only",
            "3 — Limited self-care",
            "4 — Completely disabled"
        ])
        personal_history = st.selectbox("Personal Cancer History", ["None", "Yes — Same Cancer", "Yes — Different Cancer"])
        family_history = st.selectbox("Family Cancer History", ["None", "Parent", "Sibling", "Grandparent", "Multiple Members"])

    st.divider()
    st.header("Tumor Profile")
    col4, col5 = st.columns(2)

    with col4:
        cancer_type_option = st.selectbox("Cancer Type", ALL_CANCER_TYPES + ["Other (specify below)"])
        if cancer_type_option == "Other (specify below)":
            cancer_type = st.text_input("Specify Cancer Type", placeholder="e.g. Ampullary Carcinoma")
        else:
            cancer_type = cancer_type_option
        stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.1, max_value=30.0, value=2.0)
        metastasis = st.radio("Metastasis", ["No", "Yes"])

    with col5:
        genes = st.multiselect("Mutated Genes", ALL_GENES, default=["TP53"])
        other_genes = st.text_input("Other Genes Not Listed (comma separated)", placeholder="e.g. GATA3, PIK3R1")
        if other_genes:
            extra_genes = [g.strip() for g in other_genes.split(",")]
            genes = genes + extra_genes
        mutation_burden = st.slider("Tumor Mutation Burden (mutations/Mb)", 1, 100, 20)
        msi_status = st.selectbox("MSI Status", ["MSS", "MSI-Low", "MSI-High"])
        pdl1 = st.slider("PD-L1 Expression (%)", 0, 100, 10)

    st.divider()
    st.header("Biomarkers and Lab Values")
    col6, col7 = st.columns(2)

    with col6:
        hla_type = st.selectbox("HLA Type", ["HLA-A", "HLA-B", "HLA-C", "HLA-DR", "Unknown"])
        binding_affinity = st.slider("Peptide Binding Affinity (nM)", 0.0, 500.0, 150.0)
        expression_level = st.slider("Gene Expression Level", 0.0, 10.0, 5.0)
        immunogenicity_score = st.slider("Immunogenicity Score", 0.0, 1.0, 0.5)

    with col7:
        wbc = st.number_input("WBC Count (x10/uL)", min_value=0.0, max_value=100.0, value=7.0)
        lymphocyte = st.number_input("Lymphocyte Count (x10/uL)", min_value=0.0, max_value=50.0, value=2.0)
        ldh = st.number_input("LDH Level (U/L)", min_value=0, max_value=5000, value=200)

    st.divider()

    if st.button("Predict Best Immunotherapy", type="primary"):
        if not cancer_type:
            st.error("Please enter a cancer type before predicting.")
        else:
            with st.spinner("Analyzing patient and tumor profile..."):
                prediction, results = predict_therapy(
                    cancer_type, genes, mutation_burden,
                    binding_affinity, expression_level,
                    immunogenicity_score
                )

                st.markdown("""
                    <div class='disclaimer-box'>
                        <b>Disclaimer:</b> These predictions are generated by an AI model
                        trained on research data. They do not constitute a medical diagnosis
                        or treatment recommendation.
                    </div>
                """, unsafe_allow_html=True)

                st.header("Results")
                col8, col9, col10, col11 = st.columns(4)
                col8.metric("Top Treatment", prediction[:20])
                col9.metric("Cancer Type", cancer_type)
                col10.metric("TMB", str(mutation_burden) + " mut/Mb")
                col11.metric("BMI", bmi)

                st.subheader("All Treatments Ranked")
                fig = px.bar(
                    results.head(8),
                    x="Confidence Score",
                    y="Therapy",
                    orientation="h",
                    color="Confidence Score",
                    color_continuous_scale="viridis",
                    title="Immunotherapy Match Confidence Scores"
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff", family="Inter")
                )
                st.plotly_chart(fig)

                st.subheader("Drug Information")
                st.markdown("""
                    <div class='disclaimer-box'>
                        <b>Disclaimer:</b> Drug information shown is for educational
                        reference only. Dosing, eligibility and treatment decisions
                        must be made by a licensed medical professional.
                    </div>
                """, unsafe_allow_html=True)

                for _, row in results.head(5).iterrows():
                    therapy = row['Therapy']
                    score = row['Confidence Score']
                    if therapy in DRUG_INFO:
                        drug = DRUG_INFO[therapy]
                        st.markdown(f"""
                            <div class='drug-card'>
                                <h4 style='color:#a8a8a8; margin-bottom:12px;'>{therapy}</h4>
                                <p><b>Brand Name:</b> {drug['brand']}</p>
                                <p><b>FDA Status:</b> {drug['fda']}</p>
                                <p><b>Standard Dose:</b> {drug['dose']}</p>
                                <p><b>Best For:</b> {drug['best_for']}</p>
                                <p><b>Mechanism:</b> {drug['mechanism']}</p>
                                <p><b>Side Effects:</b> {drug['side_effects']}</p>
                                <p><b>Confidence Score:</b> {score:.1%}</p>
                            </div>
                        """, unsafe_allow_html=True)

                results["Confidence Score"] = results["Confidence Score"].apply(lambda x: f"{x:.1%}")
                st.dataframe(results)

with tab2:
    st.header("About Oncovix")
    st.markdown("""
**What is Oncovix?**

Oncovix is an AI-powered precision oncology platform designed to bridge the gap between tumor genomics and personalized immunotherapy selection. By analyzing a patient's complete tumor mutation profile alongside key clinical and biological markers, Oncovix predicts which immunotherapy treatment is most likely to be effective for that specific patient.

**Why It Matters**

Cancer affects millions of people worldwide yet treatment decisions are still largely based on broad clinical guidelines rather than individual genomic data. Oncovix addresses this gap by applying machine learning to match specific tumor profiles to the most suitable immunotherapy from a curated list of 14 clinically relevant treatments spanning checkpoint inhibitors, CAR-T cell therapy, mRNA based cancer vaccines, oncolytic virus therapy and more.

**How It Works**

Oncovix takes in patient demographics, personal and family cancer history, tumor characteristics, mutated genes, biomarkers and lab values. A Random Forest Classification model trained on clinical parameters processes this data and returns a ranked list of immunotherapy options with confidence scores, drug information, FDA approval status and side effect profiles.

**Data and Technology**

Oncovix draws on treatment frameworks from The Cancer Genome Atlas (TCGA), cancer immunotherapy literature and clinical trial outcome data. It covers 30 cancer types, 40 clinically relevant genes and 14 immunotherapy options.

**Future Development**

Future versions of Oncovix will integrate real TCGA patient genomic data, HLA specific neoantigen binding prediction and validation against published clinical trial outcomes.

**Disclaimer:** Oncovix is intended for research and educational purposes only. All predictions are AI generated and should never replace the judgment of a qualified oncologist. Always consult a licensed medical professional for treatment decisions.
    """)

with tab3:
    st.header("About the Researcher")
    st.markdown("""
**Brinda Patel** is an undergraduate researcher driven by a deep commitment to advancing the frontiers of biomedical science through artificial intelligence. She built Oncovix out of a conviction that precision medicine should be accessible, intelligent and actionable, and that the gap between genomic data and clinical decision-making is a problem worth solving.

Her work sits at the intersection of computational biology, data science and drug development. She is particularly interested in how machine learning can accelerate the discovery of next generation medical treatments, from early stage research to clinical application.

She aspires to a career in biological research and development, contributing to the discovery of future therapies, driving innovation in drug development and using data analysis to push the boundaries of what modern medicine can achieve.

**Connect**

[LinkedIn](https://www.linkedin.com/in/brindapatell) | [Email](mailto:bripatel0709@gmail.com)
    """)

st.markdown("""
    <div class='footer'>
        2026 Brinda Patel. All rights reserved. Oncovix is protected under Creative Commons Attribution-NonCommercial 4.0.
        Built with Python, Scikit-learn and Streamlit. For research and educational purposes only.
    </div>
""", unsafe_allow_html=True)