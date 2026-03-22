import pandas as pd
import numpy as np

def relative_abundance(otu_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to relative abundance (%)"""
    return otu_df.div(otu_df.sum(axis=1), axis=0) * 100

def clr_transform(otu_df: pd.DataFrame) -> pd.DataFrame:
    """Centered Log-Ratio transform for compositional data"""
    pseudo = otu_df + 0.5  # pseudocount
    log_df = np.log(pseudo)
    clr = log_df.sub(log_df.mean(axis=1), axis=0)
    return clr

def filter_low_abundance(otu_df: pd.DataFrame, min_prevalence=0.1, min_abundance=0.01) -> pd.DataFrame:
    """Remove taxa present in fewer than min_prevalence fraction of samples"""
    prevalence = (otu_df > 0).mean(axis=0)
    mean_abund = relative_abundance(otu_df).mean(axis=0)
    mask = (prevalence >= min_prevalence) & (mean_abund >= min_abundance)
    return otu_df.loc[:, mask]

def preprocess_pipeline(otu_df: pd.DataFrame):
    filtered = filter_low_abundance(otu_df)
    rel_abund = relative_abundance(filtered)
    clr = clr_transform(filtered)
    return filtered, rel_abund, clr
