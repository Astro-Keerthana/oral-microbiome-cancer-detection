import streamlit as st
import pandas as pd
import numpy as np
import io

# ── Page config
st.set_page_config(
    page_title="OralBiomarker Explorer",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import modules
from modules.data_loader import (fetch_all_projects, build_otu_table_from_metadata,
                                  assign_labels, BIOPROJECTS)
from modules.diversity import (compute_alpha_diversity, plot_alpha_diversity,
                                compute_beta_diversity, plot_beta_diversity)
from modules.taxonomy import (plot_genus_abundance, plot_phylum_pie,
                               plot_heatmap_top_genera)
from modules.ml_classifier import (prepare_features, train_random_forest,
                                    plot_feature_importance, plot_roc_curve,
                                    plot_pca, get_classification_report)
from modules.report import generate_pdf_report

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
  
    st.title("OralBiomarker Explorer")
    st.markdown("**NCBI SRA Data**")
    st.markdown("---")

    st.markdown("### 📁 Data Sources")
    for bp_id, bp_name in BIOPROJECTS.items():
        st.markdown(f"- [`{bp_id}`](https://www.ncbi.nlm.nih.gov/bioproject/{bp_id}) {bp_name}")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_n_genera = st.slider("Top N Genera", 5, 20, 15)
    n_top_features = st.slider("ML Feature Importance Top N", 5, 20, 15)
    selected_group = st.selectbox("Phylum Pie Chart Group", ["OSCC", "Control"])

    st.markdown("---")
    st.markdown("### 📚 References")
    st.markdown("""
    - [PRJNA813634](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA813634)
    - [PRJNA587078](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA587078)
    - [PRJNA751046](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA751046)
    - [Oyeyemi et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10685184/)
    - [Sawant et al. 2023](https://doi.org/10.3390/pathogens12010078)
    """)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🦷 OralBiomarker Explorer")
st.markdown("""
> **Real-data OSCC oral microbiome analysis platform** powered by NCBI SRA public datasets.  
> Data sources: `PRJNA813634` · `PRJNA587078` · `PRJNA751046`
""")

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📥 Step 1 — Load Real NCBI Data")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    metadata = fetch_all_projects()
    if metadata.empty:
        st.error("⚠️ Could not fetch NCBI data. Check your internet connection.")
        st.stop()
    labels = assign_labels(metadata)
    otu_df = build_otu_table_from_metadata(metadata)
    return metadata, labels, otu_df

with st.spinner("🔄 Fetching real metadata from NCBI SRA (PRJNA813634, PRJNA587078, PRJNA751046)..."):
    metadata_df, labels, otu_df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Samples", len(metadata_df))
col2.metric("🔴 OSCC Samples", (labels == "OSCC").sum())
col3.metric("🟢 Control Samples", (labels == "Control").sum())
col4.metric("🦠 Genera Tracked", len(otu_df) - 1)

with st.expander("📋 View Raw NCBI Metadata", expanded=False):
    st.dataframe(metadata_df, use_container_width=True)

# ─────────────────────────────────────────────
# TAXONOMY
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🦠 Step 2 — Taxonomy Analysis")

tab1, tab2, tab3 = st.tabs(["📊 Genus Abundance", "🥧 Phylum Composition", "🔥 Heatmap"])

with tab1:
    fig_genus = plot_genus_abundance(otu_df, labels, top_n=top_n_genera)
    st.plotly_chart(fig_genus, use_container_width=True)
    st.caption("Genera confirmed in OSCC microbiome studies: Oyeyemi et al. 2023 (PMC10685184), Sawant et al. 2023 (PRJNA751046)")

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie_oscc = plot_phylum_pie(otu_df, labels, group="OSCC")
        st.plotly_chart(fig_pie_oscc, use_container_width=True)
    with col_b:
        fig_pie_ctrl = plot_phylum_pie(otu_df, labels, group="Control")
        st.plotly_chart(fig_pie_ctrl, use_container_width=True)

with tab3:
    fig_heat = plot_heatmap_top_genera(otu_df, labels, top_n=top_n_genera)
    st.plotly_chart(fig_heat, use_container_width=True)

# ─────────────────────────────────────────────
# DIVERSITY
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🧬 Step 3 — Diversity Analysis")

alpha_df = compute_alpha_diversity(otu_df, labels)

col_div1, col_div2 = st.columns(2)
with col_div1:
    fig_alpha = plot_alpha_diversity(alpha_df)
    st.plotly_chart(fig_alpha, use_container_width=True)

with col_div2:
    st.markdown("#### 📊 Alpha Diversity Summary")
    summary = alpha_df.groupby("Group")[["Shannon", "Simpson", "Observed_OTUs"]].agg(["mean", "std"])
    st.dataframe(summary.round(3), use_container_width=True)

st.markdown("#### 🌐 Beta Diversity (Bray-Curtis)")
with st.spinner("Computing Bray-Curtis dissimilarity matrix..."):
    bc_matrix = compute_beta_diversity(otu_df)
    fig_beta = plot_beta_diversity(bc_matrix, labels)
    st.plotly_chart(fig_beta, use_container_width=True)

# ─────────────────────────────────────────────
# MACHINE LEARNING
# ─────────────────────────────────────────────
st.markdown("---")
st.header("🤖 Step 4 — ML Biomarker Classification")

with st.spinner("Training Random Forest classifier..."):
    X, y, le, feature_names = prepare_features(otu_df, labels)
    clf, cv_scores = train_random_forest(X, y)

col_ml1, col_ml2, col_ml3 = st.columns(3)
col_ml1.metric("🎯 Mean AUC (5-fold CV)", f"{cv_scores.mean():.3f}")
col_ml2.metric("📉 AUC Std Dev", f"{cv_scores.std():.3f}")
col_ml3.metric("🌲 Trees in Forest", "200")

tab_ml1, tab_ml2, tab_ml3 = st.tabs(["🏆 Feature Importance", "📈 ROC Curve", "🔵 PCA Plot"])

with tab_ml1:
    fig_fi = plot_feature_importance(clf, feature_names, top_n=n_top_features)
    st.plotly_chart(fig_fi, use_container_width=True)

with tab_ml2:
    fig_roc = plot_roc_curve(clf, X, y)
    st.plotly_chart(fig_roc, use_container_width=True)

with tab_ml3:
    fig_pca = plot_pca(X, y, le)
    st.plotly_chart(fig_pca, use_container_width=True)

with st.expander("📋 Full Classification Report"):
    clf_report = get_classification_report(clf, X, y, le)
    st.code(clf_report)

# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
st.markdown("---")
st.header("📄 Step 5 — Download PDF Report")

if st.button("🖨️ Generate & Download PDF Report", type="primary"):
    with st.spinner("Generating PDF..."):
        clf_report_str = get_classification_report(clf, X, y, le)
        pdf_bytes = generate_pdf_report(
            metadata_df=metadata_df,
            alpha_df=alpha_df,
            cv_scores=cv_scores,
            clf_report=clf_report_str,
            bioproject_ids=list(BIOPROJECTS.keys())
        )
    st.download_button(
        label="⬇️ Download Report PDF",
        data=pdf_bytes,
        file_name="OralBiomarker_OSCC_Report.pdf",
        mime="application/pdf"
    )
    st.success("✅ Report generated successfully!")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:12px'>
OralBiomarker Explorer · Real data from NCBI BioProjects 
<a href='https://www.ncbi.nlm.nih.gov/bioproject/PRJNA813634'>PRJNA813634</a> · 
<a href='https://www.ncbi.nlm.nih.gov/bioproject/PRJNA587078'>PRJNA587078</a> · 
<a href='https://www.ncbi.nlm.nih.gov/bioproject/PRJNA751046'>PRJNA751046</a><br>
References: Oyeyemi et al. 2023 (PMC10685184) · Sawant et al. 2023 (Pathogens)
</div>
""", unsafe_allow_html=True)
