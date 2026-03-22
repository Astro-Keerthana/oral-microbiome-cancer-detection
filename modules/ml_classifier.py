import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def prepare_features(otu_df: pd.DataFrame, labels: pd.Series):
    """Prepare feature matrix X and encoded labels y."""
    genus_cols = [c for c in otu_df.columns if c != "Phylum"]
    X = otu_df[genus_cols].T.values.astype(float)
    X = X[:len(labels)]

    # Relative abundance normalization
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums

    le = LabelEncoder()
    y = le.fit_transform(labels.values)
    return X, y, le, genus_cols


def train_random_forest(X, y):
    """Train Random Forest with 5-fold cross-validation."""
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    clf.fit(X, y)
    return clf, scores


def plot_feature_importance(clf, feature_names: list, top_n: int = 15) -> go.Figure:
    """Bar chart of top N most important genera for classification."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = [importances[i] for i in indices]

    fig = go.Figure(go.Bar(
        x=top_scores[::-1], y=top_features[::-1],
        orientation="h",
        marker=dict(
            color=top_scores[::-1],
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Importance")
        )
    ))
    fig.update_layout(
        title=f"🤖 Top {top_n} Biomarker Genera — Random Forest Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Genus",
        template="plotly_white",
        height=500
    )
    return fig


def plot_roc_curve(clf, X, y) -> go.Figure:
    """Plot ROC curve with AUC score."""
    y_prob = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"ROC (AUC = {roc_auc:.3f})",
        line=dict(color="#E74C3C", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="gray", dash="dash")
    ))
    fig.update_layout(
        title="📈 ROC Curve — OSCC vs Control Classifier",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white", height=450,
        legend=dict(x=0.6, y=0.1)
    )
    return fig


def plot_pca(X, y, le) -> go.Figure:
    """2D PCA scatter plot colored by group."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    labels_decoded = le.inverse_transform(y)
    df_pca = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Group": labels_decoded
    })

    fig = px.scatter(
        df_pca, x="PC1", y="PC2", color="Group",
        color_discrete_map={"OSCC": "#E74C3C", "Control": "#2ECC71"},
        title=f"🔵 PCA — Microbiome Clustering (PC1: {pca.explained_variance_ratio_[0]*100:.1f}% | PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)",
        template="plotly_white", height=480,
        symbol="Group", opacity=0.8
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig


def get_classification_report(clf, X, y, le) -> str:
    """Return classification report as string."""
    y_pred = clf.predict(X)
    return classification_report(y, y_pred, target_names=le.classes_)
