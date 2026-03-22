from fpdf import FPDF
import pandas as pd
import numpy as np
from datetime import datetime
import io

class OSCCReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(231, 76, 60)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "OralBiomarker Explorer — OSCC Microbiome Report", 
                  align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Data: NCBI BioProjects PRJNA813634, PRJNA587078, PRJNA751046", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(44, 62, 80)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_table(self, headers: list, rows: list):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(189, 195, 199)
        col_w = 180 // len(headers)
        for h in headers:
            self.cell(col_w, 7, str(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(245, 245, 245) if fill else self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_w, 6, str(cell)[:25], border=1, fill=fill, align="C")
            self.ln()
        self.ln(4)

def generate_pdf_report(metadata_df: pd.DataFrame,
                         alpha_df: pd.DataFrame,
                         cv_scores: np.ndarray,
                         clf_report: str,
                         bioproject_ids: list) -> bytes:
    """Generate a full PDF report and return as bytes."""
    pdf = OSCCReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Section 1: Study Overview
    pdf.section_title("1. Study Overview")
    pdf.body_text(
        "This report summarizes the oral microbiome analysis of Oral Squamous Cell Carcinoma (OSCC) "
        "patients versus healthy controls, using publicly available 16S rRNA amplicon sequencing data "
        "from NCBI SRA. Data was retrieved from BioProjects: " + ", ".join(bioproject_ids) + ".\n\n"
        "Key references:\n"
        "  • Oyeyemi et al. (2023) Heliyon — PMC10685184\n"
        "  • Sawant et al. (2023) Pathogens — PRJNA751046\n"
        "  • Sun Yat-sen University (2022) — PRJNA813634\n"
        "  • Zhengzhou University (2019) — PRJNA587078"
    )

    # ── Section 2: Dataset Summary
    pdf.section_title("2. Dataset Summary")
    n_oscc = len(metadata_df[metadata_df.get("label", pd.Series()) == "OSCC"]) if "label" in metadata_df.columns else "N/A"
    n_ctrl = len(metadata_df[metadata_df.get("label", pd.Series()) == "Control"]) if "label" in metadata_df.columns else "N/A"
    pdf.body_text(
        f"Total samples retrieved: {len(metadata_df)}\n"
        f"OSCC samples: {n_oscc}\n"
        f"Control samples: {n_ctrl}\n"
        f"BioProjects analyzed: {', '.join(bioproject_ids)}"
    )

    headers = ["Sample", "Group", "Shannon", "Simpson", "Observed OTUs"]
    rows = []
    for _, row in alpha_df.head(20).iterrows():
        rows.append([
            str(row.get("Sample", ""))[:15],
            str(row.get("Group", "")),
            f"{row.get('Shannon', 0):.3f}",
            f"{row.get('Simpson', 0):.3f}",
            str(row.get("Observed_OTUs", ""))
        ])
    pdf.add_table(headers, rows)

    # ── Section 3: Alpha Diversity
    pdf.section_title("3. Alpha Diversity Results")
    oscc_sh = alpha_df[alpha_df["Group"] == "OSCC"]["Shannon"].mean()
    ctrl_sh = alpha_df[alpha_df["Group"] == "Control"]["Shannon"].mean()
    pdf.body_text(
        f"Mean Shannon Index — OSCC: {oscc_sh:.3f} | Control: {ctrl_sh:.3f}\n"
        "Lower Shannon diversity in OSCC samples is consistent with published literature "
        "showing microbial dysbiosis in oral cancer (Oyeyemi et al. 2023, PMC10685184)."
    )

    # ── Section 4: ML Results
    pdf.section_title("4. Machine Learning Classification Results")
    pdf.body_text(
        f"Model: Random Forest Classifier (200 trees, 5-fold stratified CV)\n"
        f"Cross-validated AUC scores: {', '.join([f'{s:.3f}' for s in cv_scores])}\n"
        f"Mean AUC ± SD: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n\n"
        "Classification Report:\n" + clf_report
    )

    # ── Section 5: References
    pdf.section_title("5. Data Sources & References")
    pdf.body_text(
        "1. NCBI BioProject PRJNA813634 — Oral Microbiome of Oral Cancer, Sun Yat-sen University (2022)\n"
        "   https://www.ncbi.nlm.nih.gov/bioproject/PRJNA813634\n\n"
        "2. NCBI BioProject PRJNA587078 — Human Oral Microbiome, Zhengzhou University (2019)\n"
        "   https://www.ncbi.nlm.nih.gov/bioproject/PRJNA587078\n\n"
        "3. NCBI BioProject PRJNA751046 — Tobacco Chewers & Oral Cancer India\n"
        "   Sawant et al. (2023) Pathogens. doi:10.3390/pathogens12010078\n\n"
        "4. Oyeyemi BF et al. (2023) Microbiome analysis of saliva from OSCC patients.\n"
        "   Heliyon. doi:10.1016/j.heliyon.2023.e21773 — PMC10685184"
    )

    return bytes(pdf.output())
