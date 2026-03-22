import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import braycurtis
from scipy.stats import mannwhitneyu

def shannon_entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    total = counts.sum()
    p = counts / total
    return float(-np.sum(p * np.log(p)))

def simpson_index(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    total = counts.sum()
    p = counts / total
    return float(1 - np.sum(p ** 2))

def observed_otus(counts: np.ndarray) -> int:
    return int(np.sum(counts > 0))

def compute_alpha_diversity(otu_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    genus_cols = [c for c in otu_df.columns if c != "Phylum"]
    results = []
    for sample in genus_cols:
        counts = otu_df[sample].values.astype(float)
        results.append({
            "Sample": sample,
            "Shannon": shannon_entropy(counts),
            "Simpson": simpson_index(counts),
            "Observed_OTUs": observed_otus(counts),
        })
    alpha_df = pd.DataFrame(results)
    alpha_df["Group"] = labels.values[:len(alpha_df)]
    return alpha_df

def plot_alpha_diversity(alpha_df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        alpha_df, x="Group", y="Shannon", color="Group",
        box=True, points="all",
        color_discrete_map={"OSCC": "#E74C3C", "Control": "#2ECC71"},
        title="🧬 Alpha Diversity — Shannon Index (OSCC vs Control)",
        labels={"Shannon": "Shannon Entropy", "Group": "Sample Group"},
        template="plotly_white"
    )

    oscc_vals = alpha_df[alpha_df["Group"] == "OSCC"]["Shannon"].dropna()
    ctrl_vals = alpha_df[alpha_df["Group"] == "Control"]["Shannon"].dropna()
    if len(oscc_vals) > 1 and len(ctrl_vals) > 1:
        stat, pval = mannwhitneyu(oscc_vals, ctrl_vals, alternative="two-sided")
        fig.add_annotation(
            x=0.5, y=1.05, xref="paper", yref="paper",
            text=f"Mann-Whitney U p = {pval:.4f}",
            showarrow=False, font=dict(size=12)
        )
    fig.update_layout(height=500)
    return fig

def compute_beta_diversity(otu_df: pd.DataFrame) -> pd.DataFrame:

    genus_cols = [c for c in otu_df.columns if c != "Phylum"]
    matrix = otu_df[genus_cols].values.T.astype(float)
    n = len(genus_cols)
    bc_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bc_matrix[i, j] = braycurtis(matrix[i], matrix[j])
    return pd.DataFrame(bc_matrix, index=genus_cols, columns=genus_cols)

def plot_beta_diversity(bc_matrix: pd.DataFrame, labels: pd.Series) -> go.Figure:

    label_colors = ["#E74C3C" if l == "OSCC" else "#2ECC71"
                    for l in labels.values[:len(bc_matrix)]]
    fig = go.Figure(data=go.Heatmap(
        z=bc_matrix.values,
        x=bc_matrix.columns.tolist(),
        y=bc_matrix.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="Bray-Curtis"),
    ))
    fig.update_layout(
        title=" Beta Diversity — Bray-Curtis Dissimilarity Matrix",
        height=600,
        template="plotly_white",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    return fig
