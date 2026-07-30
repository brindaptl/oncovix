import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

ALL_CANCER_TYPES = [
    'Breast', 'Lung', 'Colorectal', 'Melanoma', 'Bladder',
    'Prostate', 'Ovarian', 'Pancreatic', 'Leukemia', 'Lymphoma',
    'Brain (Glioblastoma)', 'Liver', 'Kidney', 'Thyroid',
    'Stomach', 'Cervical', 'Uterine', 'Testicular',
    'Bone (Osteosarcoma)', 'Soft Tissue Sarcoma',
    'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
    'Merkel Cell Carcinoma', 'Cutaneous T-Cell Lymphoma',
    'Multiple Myeloma', 'Mesothelioma', 'Esophageal',
    'Head and Neck', 'Adrenal', 'Gallbladder'
]

ALL_GENES = [
    'TP53', 'BRCA1', 'BRCA2', 'EGFR', 'KRAS', 'PIK3CA',
    'PTEN', 'RB1', 'BRAF', 'NRAS', 'CDKN2A', 'KIT',
    'MC1R', 'TERT', 'MYC', 'VEGFR', 'ALK', 'RET',
    'FGFR1', 'FGFR2', 'IDH1', 'IDH2', 'ARID1A',
    'POLE', 'MLH1', 'MSH2', 'APC', 'VHL', 'MET',
    'HER2', 'AR', 'JAK2', 'NPM1', 'FLT3', 'DNMT3A',
    'CDKN2B', 'SMAD4', 'NOTCH1', 'SF3B1', 'ATM'
]

DRUG_INFO = {
    'PD-1 Inhibitor (Pembrolizumab)': {
        'brand': 'Keytruda',
        'fda': 'FDA Approved',
        'dose': '200mg every 3 weeks IV',
        'best_for': 'High TMB, MSI-High, PD-L1 >50%',
        'mechanism': 'Blocks PD-1 receptor allowing T-cells to attack tumor',
        'side_effects': 'Fatigue, rash, colitis, pneumonitis'
    },
    'PD-L1 Inhibitor (Atezolizumab)': {
        'brand': 'Tecentriq',
        'fda': 'FDA Approved',
        'dose': '1200mg every 3 weeks IV',
        'best_for': 'PD-L1 positive tumors, NSCLC, bladder cancer',
        'mechanism': 'Blocks PD-L1 on tumor cells restoring immune attack',
        'side_effects': 'Fatigue, nausea, decreased appetite'
    },
    'CTLA-4 Inhibitor (Ipilimumab)': {
        'brand': 'Yervoy',
        'fda': 'FDA Approved',
        'dose': '3mg/kg every 3 weeks IV',
        'best_for': 'Melanoma, high mutation burden tumors',
        'mechanism': 'Blocks CTLA-4 boosting T-cell activation',
        'side_effects': 'Colitis, hepatitis, endocrinopathy'
    },
    'CAR-T Cell Therapy': {
        'brand': 'Kymriah / Yescarta',
        'fda': 'FDA Approved (select cancers)',
        'dose': 'Single infusion, patient specific',
        'best_for': 'B-cell lymphoma, ALL, multiple myeloma',
        'mechanism': 'Engineered T-cells targeting tumor antigens',
        'side_effects': 'Cytokine release syndrome, neurotoxicity'
    },
    'Cancer Vaccine (mRNA-based)': {
        'brand': 'Investigational',
        'fda': 'Breakthrough Therapy Designation',
        'dose': 'Patient specific dosing',
        'best_for': 'High neoantigen burden, melanoma, NSCLC',
        'mechanism': 'mRNA instructs immune system to target tumor neoantigens',
        'side_effects': 'Injection site reaction, fatigue, fever'
    },
    'BRAF Inhibitor (Vemurafenib)': {
        'brand': 'Zelboraf',
        'fda': 'FDA Approved',
        'dose': '960mg twice daily oral',
        'best_for': 'BRAF V600E mutated melanoma',
        'mechanism': 'Blocks mutated BRAF protein driving tumor growth',
        'side_effects': 'Skin rash, joint pain, photosensitivity'
    },
    'MEK Inhibitor (Trametinib)': {
        'brand': 'Mekinist',
        'fda': 'FDA Approved',
        'dose': '2mg once daily oral',
        'best_for': 'BRAF mutated melanoma, NSCLC',
        'mechanism': 'Blocks MEK protein in BRAF pathway',
        'side_effects': 'Rash, diarrhea, lymphedema'
    },
    'Oncolytic Virus Therapy (T-VEC)': {
        'brand': 'Imlygic',
        'fda': 'FDA Approved',
        'dose': 'Intralesional injection every 2 weeks',
        'best_for': 'Unresectable melanoma, accessible lesions',
        'mechanism': 'Modified herpes virus selectively kills tumor cells',
        'side_effects': 'Flu-like symptoms, injection site reactions'
    },
    'Targeted Therapy (Nivolumab)': {
        'brand': 'Opdivo',
        'fda': 'FDA Approved',
        'dose': '240mg every 2 weeks IV',
        'best_for': 'Melanoma, NSCLC, RCC, colorectal',
        'mechanism': 'Blocks PD-1 restoring anti-tumor immunity',
        'side_effects': 'Fatigue, rash, immune mediated reactions'
    },
    'Bispecific Antibody Therapy': {
        'brand': 'Blincyto / Tecvayli',
        'fda': 'FDA Approved (select cancers)',
        'dose': 'Continuous IV infusion, cycle based',
        'best_for': 'B-cell ALL, multiple myeloma',
        'mechanism': 'Bridges T-cells to tumor cells for direct killing',
        'side_effects': 'Cytokine release syndrome, neurological events'
    },
    'Tumor Infiltrating Lymphocyte Therapy': {
        'brand': 'Amtagvi',
        'fda': 'FDA Approved',
        'dose': 'Single infusion after lymphodepletion',
        'best_for': 'Unresectable melanoma',
        'mechanism': 'Expanded tumor fighting T-cells reinfused into patient',
        'side_effects': 'Cytokine release syndrome, prolonged cytopenias'
    },
    'NK Cell Therapy': {
        'brand': 'Investigational',
        'fda': 'Clinical Trials',
        'dose': 'Patient specific IV infusion',
        'best_for': 'Hematologic malignancies, solid tumors',
        'mechanism': 'Natural killer cells target and destroy tumor cells',
        'side_effects': 'Infusion reactions, cytokine release'
    },
    'Adoptive Cell Transfer': {
        'brand': 'Investigational',
        'fda': 'Clinical Trials',
        'dose': 'Patient specific',
        'best_for': 'Solid tumors with high mutation burden',
        'mechanism': 'Patient immune cells expanded and reinfused',
        'side_effects': 'Cytokine release syndrome, autoimmunity'
    },
    'Checkpoint Blockade Combination': {
        'brand': 'Opdualag / Nivolumab + Ipilimumab',
        'fda': 'FDA Approved',
        'dose': 'Combination dosing every 3-4 weeks',
        'best_for': 'Melanoma, RCC, NSCLC with high TMB',
        'mechanism': 'Dual checkpoint blockade maximizes T-cell response',
        'side_effects': 'Higher rates of immune mediated adverse events'
    }
}

def generate_sample_data():
    np.random.seed(42)
    n_samples = 500
    data = {
        'gene_name': np.random.choice(ALL_GENES, n_samples),
        'cancer_type': np.random.choice(ALL_CANCER_TYPES, n_samples),
        'binding_affinity': np.random.uniform(0, 500, n_samples),
        'mutation_burden': np.random.randint(1, 100, n_samples),
        'expression_level': np.random.uniform(0, 10, n_samples),
        'immunogenicity_score': np.random.uniform(0, 1, n_samples)
    }
    df = pd.DataFrame(data)

    def assign_therapy(row):
        if row['immunogenicity_score'] >= 0.85:
            return 'Cancer Vaccine (mRNA-based)'
        elif row['immunogenicity_score'] >= 0.75:
            return 'CAR-T Cell Therapy'
        elif row['immunogenicity_score'] >= 0.65:
            return 'PD-1 Inhibitor (Pembrolizumab)'
        elif row['immunogenicity_score'] >= 0.55:
            return 'Tumor Infiltrating Lymphocyte Therapy'
        elif row['immunogenicity_score'] >= 0.45:
            return 'Checkpoint Blockade Combination'
        elif row['immunogenicity_score'] >= 0.35:
            return 'BRAF Inhibitor (Vemurafenib)'
        elif row['immunogenicity_score'] >= 0.25:
            return 'PD-L1 Inhibitor (Atezolizumab)'
        elif row['immunogenicity_score'] >= 0.15:
            return 'MEK Inhibitor (Trametinib)'
        elif row['immunogenicity_score'] >= 0.05:
            return 'CTLA-4 Inhibitor (Ipilimumab)'
        else:
            return 'Bispecific Antibody Therapy'

    df['recommended_therapy'] = df.apply(assign_therapy, axis=1)
    return df

def train_model():
    df = generate_sample_data()
    le_cancer = LabelEncoder()
    le_gene = LabelEncoder()
    df['cancer_encoded'] = le_cancer.fit_transform(df['cancer_type'])
    df['gene_encoded'] = le_gene.fit_transform(df['gene_name'])
    features = ['binding_affinity', 'mutation_burden', 'expression_level',
                'immunogenicity_score', 'cancer_encoded', 'gene_encoded']
    X = df[features]
    y = df['recommended_therapy']
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, le_cancer, le_gene

def predict_therapy(cancer_type, genes, mutation_burden, binding_affinity,
                   expression_level, immunogenicity_score):
    model, le_cancer, le_gene = train_model()
    try:
        cancer_encoded = le_cancer.transform([cancer_type])[0]
    except:
        cancer_encoded = 0
    try:
        gene_encoded = le_gene.transform([genes[0] if genes else 'TP53'])[0]
    except:
        gene_encoded = 0
    input_data = pd.DataFrame([{
        'binding_affinity': binding_affinity,
        'mutation_burden': mutation_burden,
        'expression_level': expression_level,
        'immunogenicity_score': immunogenicity_score,
        'cancer_encoded': cancer_encoded,
        'gene_encoded': gene_encoded
    }])
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    results = pd.DataFrame({
        'Therapy': classes,
        'Confidence Score': probabilities
    }).sort_values('Confidence Score', ascending=False)
    return prediction, results
# ============================================
# Oncovix - AI Cancer Immunotherapy Predictor
# Created by Brinda Patel, 2026
# All rights reserved
# Creative Commons Attribution-NonCommercial 4.0
# ============================================