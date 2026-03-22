# oral-microbiome-cancer-detection

# OralBiomarker Explorer

Real-data OSCC oral microbiome analysis platform using public NCBI SRA datasets.

## Data Sources
| BioProject | Description | Link |
|---|---|---|
| PRJNA813634 | Oral Microbiome of Oral Cancer (Sun Yat-sen Univ.) | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA813634) |
| PRJNA587078 | Human Oral Microbiome (Zhengzhou Univ.) | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA587078) |
| PRJNA751046 | Tobacco Chewers & Oral Cancer India | [NCBI](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA751046) |

## Run Locally
```bash
git clone https://github.com/astro-keerthana/oral-microbiome-cancer-detection
cd OralBiomarker-Explorer
pip install -r requirements.txt
streamlit run app.py
