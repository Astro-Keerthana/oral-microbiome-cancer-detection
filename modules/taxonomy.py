import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_genus_abundance(otu_df: pd.DataFrame, labels: pd.Series,
                          top_n: int = 15) -> go.Figure:
    genus_cols = [c for c in otu_df.columns if c != "Phylum"]

    oscc_idx = [i for i, l in enumerate(labels) if l == "OSCC"]
    ctrl_idx = [i for i, l in enumerate(labels) if l == "Control"]

    oscc_samples = [genus_cols[i] for i in oscc_idx if i < len(genus_cols)]
    ctrl_samples = [genus_cols[i] for i in ctrl_idx if i < len(genus_cols)]

    oscc_mean = otu_df[oscc_samples].mean(axis=1) if oscc_samples else pd.Series(0, index=otu_df.index)
    ctrl_mean = otu_df[ctrl_samples].mean(axis=1) if ctrl_samples else pd.Series(0, index=otu_df.index)

    combined = pd.DataFrame({
        "Genus": otu_df.index,
        "OSCC": oscc_mean.values,
        "Control": ctrl_mean.values
    }).sort_values("OSCC", ascending=False).head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="OSCC", x=combined["Genus"],
                          y=combined["OSCC"], marker_color="#E74C3C"))
    fig.add_trace(go.Bar(name="Control", x=combined["Genus"],
                          y=combined["Control"], marker_color="#2ECC71"))
    fig.update_layout(
        barmode="group",
        title=f"🦠 Top {top_n} Genera — Mean Abundance (OSCC vs Control)",
        xaxis_title="Genus", yaxis_title="Mean Read Count",
        template="plotly_white", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

def plot_phylum_pie(otu_df: pd.DataFrame, labels: pd.Series,
                    group: str = "OSCC") -> go.Figure:

    genus_cols = [c for c in otu_df.columns if c != "Phylum"]
    group_samples = [genus_cols[i] for i, l in enumerate(labels)
                     if l == group and i < len(genus_cols)]

    if not group_samples:
        return go.Figure()

    phylum_sum = otu_df.groupby("Phylum")[group_samples].sum().sum(axis=1)
    phylum_df = phylum_sum.reset_index()
    phylum_df.columns = ["Phylum", "Total"]

    fig = px.pie(
        phylum_df, names="Phylum", values="Total",
        title=f"🔬 Phylum Composition — {group}",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.35
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white", height=450)
    return fig

def plot_heatmap_top_genera(otu_df: pd.DataFrame, labels: pd.Series,
                             top_n: int = 20) -> go.Figure:
   
    genus_cols = [c for c in otu_df.columns if c != "Phylum"]
    sub = otu_df[genus_cols].copy()

    top_genera = sub.sum(axis=1).nlargest(top_n).index
    sub_top = sub.loc[top_genera]

    # Normalize per sample (relative abundance)
    sub_norm = sub_top.div(sub_top.sum(axis=0), axis=1) * 100

    group_labels = [f"{s}<br>({labels.iloc[i] if i < len(labels) else '?'})"
                    for i, s in enumerate(genus_cols)]

    fig = go.Figure(data=go.Heatmap(
        z=sub_norm.values,
        x=group_labels,
        y=sub_norm.index.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="Rel. Abundance (%)"),
    ))
    fig.update_layout(
        title=f"🔥 Top {top_n} Genera Heatmap — Relative Abundance (%)",
        height=600, template="plotly_white",
        xaxis=dict(showticklabels=False),
        yaxis=dict(tickfont=dict(size=11))
    )
    return fig
