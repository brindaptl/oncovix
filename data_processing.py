import pandas as pd
import numpy as np

def generate_sample_data():
    """
    Generates sample neoantigen data for demonstration
    In real version this connects to TCGA database
    """
    np.random.seed(42)
    n_samples = 100

    data = {
        'mutation_id': [f'MUT_{i}' for i in range(n_samples)],
        'gene_name': np.random.choice([
            'TP53', 'BRCA1', 'BRCA2', 'EGFR',
            'KRAS', 'PIK3CA', 'PTEN', 'RB1'
        ], n_samples),
        'cancer_type': np.random.choice([
            'Breast', 'Lung', 'Colorectal',
            'Melanoma', 'Bladder'
        ], n_samples),
        'binding_affinity': np.random.uniform(0, 500, n_samples),
        'mutation_burden': np.random.randint(1, 100, n_samples),
        'expression_level': np.random.uniform(0, 10, n_samples),
        'immunogenicity_score': np.random.uniform(0, 1, n_samples)
    }

    df = pd.DataFrame(data)
    return df

def process_mutations(cancer_type, genes, mutation_burden):
    """
    Process user input and prepare it for prediction
    """
    data = {
        'cancer_type': [cancer_type],
        'mutation_burden': [mutation_burden],
        'gene_count': [len(genes)],
        'has_TP53': [1 if 'TP53' in genes else 0],
        'has_BRCA1': [1 if 'BRCA1' in genes else 0],
        'has_BRCA2': [1 if 'BRCA2' in genes else 0],
        'has_EGFR': [1 if 'EGFR' in genes else 0],
        'has_KRAS': [1 if 'KRAS' in genes else 0],
    }
    return pd.DataFrame(data)