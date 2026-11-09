"""
Pneumonia Detection Dashboard — Streamlit Cloud version
Hosted at: https://share.streamlit.io
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
PNG  = os.path.join(BASE, "assets", "clusters_pca.png")
PDF  = os.path.join(BASE, "assets", "Pneumonia_Detection_Report.pdf")
PCA_PARQUET = os.path.join(BASE, "data", "clustering", "pca_sample.parquet")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection — MMDS Project",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    background: linear-gradient(135deg,#1B2631,#2E4057);
    color:white; padding:28px 36px; border-radius:12px; margin-bottom:24px;
}
.main-title h1{color:white!important;margin:0;font-size:2rem;}
.main-title p{color:#AEB6BF;margin:6px 0 0;font-size:1rem;}
.metric-card{background:white;border-radius:10px;padding:20px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center;}
.metric-card .val{font-size:2.2rem;font-weight:700;color:#2E4057;}
.metric-card .lbl{font-size:0.82rem;color:#7F8C8D;margin-top:2px;}
.section-header{font-size:1.1rem;font-weight:700;color:#2E4057;
    border-left:4px solid #E74C3C;padding-left:10px;margin:20px 0 12px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 Navigation")
    page = st.radio("", [
        "🏠 Overview",
        "📊 Classification Results",
        "🔵 Clustering Results",
        "🖼️ PCA Visualisation",
        "📁 Dataset & Pipeline",
        "📄 Download Report",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset:** NIH ChestX-ray14")
    st.markdown("**Images:** 112,120")
    st.markdown("**Features:** DenseNet121 (1,024-d)")
    st.markdown("**Framework:** Apache Spark MLlib")
    st.markdown("**Best Model:** GBT (F1 = 0.913)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">
  <h1>🫁 Large-Scale Pneumonia Detection from Chest X-Rays</h1>
  <p>Mining of Massive Datasets (MMDS) Project &nbsp;|&nbsp;
     DenseNet121 Feature Extraction &nbsp;+&nbsp; Apache Spark MLlib
     Classification &amp; Clustering &nbsp;|&nbsp; 112,120 NIH Chest X-Rays</p>
</div>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
df_cls = pd.DataFrame({
    "Model":     ["Logistic Regression", "Random Forest", "GBT"],
    "AUC":       [0.600, 0.591, 0.574],
    "Accuracy":  [0.692, 0.759, 0.873],
    "F1":        [0.799, 0.844, 0.913],
    "Precision": [0.962, 0.960, 0.959],
    "Recall":    [0.692, 0.759, 0.873],
})
df_sil = pd.DataFrame({
    "k":          [2,      3,      4,      5,      6,      8],
    "Silhouette": [0.1529, 0.0815, 0.0800, 0.0637, 0.0634, 0.0385],
})
clrs_bar = ["#3498DB","#2ECC71","#E74C3C"]

# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5],[
        ("112,120","Total X-Ray Images"),("1,024","DenseNet121 Features"),
        ("87.3%","Best Accuracy (GBT)"),("0.913","Best F1 (GBT)"),("k=2","Best Clusters"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div>'
                     f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Pipeline Architecture</div>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    stages=[
        ("1️⃣ Feature Extraction","DenseNet121 (ImageNet)\n112,120 PNGs → 1,024-d\nIncremental archives\n839 MB Parquet","#3498DB"),
        ("2️⃣ Spark DataFrame","Apache Spark 4.3\nParquet → DataFrame\n112,120 × 1,024 cols\nClass weighting","#2ECC71"),
        ("3️⃣ Classification","Spark MLlib\nLogistic Regression\nRandom Forest\nGradient-Boosted Trees","#E74C3C"),
        ("4️⃣ Clustering","Spark MLlib\nStandardScaler+KMeans\nk=2…8, Silhouette\nPCA 2-D visualisation","#9B59B6"),
    ]
    for col,(title,desc,clr) in zip(cols,stages):
        col.markdown(f'<div style="background:{clr};color:white;border-radius:10px;'
                     f'padding:16px;text-align:center;height:185px">'
                     f'<b style="font-size:0.95rem">{title}</b>'
                     f'<hr style="border-color:rgba(255,255,255,0.3);margin:8px 0">'
                     f'<p style="font-size:0.78rem;margin:0;white-space:pre-line">{desc}</p>'
                     f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Dataset Class Distribution</div>',
                unsafe_allow_html=True)
    dc1,dc2 = st.columns([1,2])
    with dc1:
        fig = px.pie(values=[110689,1431],names=["Normal","Pneumonia"],
                     color_discrete_sequence=["#3498DB","#E74C3C"],hole=0.45)
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),height=260,
                          legend=dict(orientation="h",y=-0.15),
                          annotations=[dict(text="112,120",x=0.5,y=0.5,
                                            font_size=16,showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
    with dc2:
        st.markdown("""
        | Class | Count | % |
        |---|---|---|
        | Normal | 110,689 | **98.7%** |
        | Pneumonia | 1,431 | **1.3%** |
        | **Total** | **112,120** | 100% |

        > ⚠️ **77:1 class imbalance** — handled with inverse-frequency weighting in Spark MLlib.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Classification Results":
    st.markdown("## 📊 Classification Results")
    st.markdown("Three Spark MLlib classifiers trained on **86,524** images, "
                "evaluated on **25,596** test images (official NIH split).")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🏆 Best Model",     "GBT")
    m2.metric("✅ Best Accuracy",  "87.3%")
    m3.metric("📈 Best F1-Score",  "0.913")
    m4.metric("🎯 Best AUC",       "0.600 (LR)")

    st.markdown("---")
    ta,tb = st.columns([1.2,1])
    with ta:
        st.markdown("### Model Comparison")
        disp = df_cls.copy()
        disp["Accuracy"] = disp["Accuracy"].apply(lambda x:f"{x*100:.1f}%")
        for c in ["AUC","F1","Precision","Recall"]:
            disp[c] = disp[c].apply(lambda x:f"{x:.3f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.info("**GBT** achieves the highest accuracy and F1. "
                "**Logistic Regression** has the highest AUC — better at ranking "
                "Pneumonia cases. High precision (≥0.959) reflects the dominant Normal class.")
    with tb:
        st.markdown("### Grouped Bar Chart")
        fig = go.Figure()
        for i,row in df_cls.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],x=["AUC","Accuracy","F1","Precision","Recall"],
                y=[row["AUC"],row["Accuracy"],row["F1"],row["Precision"],row["Recall"]],
                marker_color=clrs_bar[i]))
        fig.update_layout(barmode="group",height=320,
                          yaxis=dict(range=[0,1.05],title="Score"),
                          legend=dict(orientation="h",y=-0.3),
                          margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Radar Chart")
    cats = ["AUC","Accuracy","F1","Precision","Recall"]
    fig2 = go.Figure()
    for i,row in df_cls.iterrows():
        vals = [row[c] for c in cats]+[row[cats[0]]]
        fig2.add_trace(go.Scatterpolar(
            r=vals,theta=cats+[cats[0]],fill="toself",
            name=row["Model"],opacity=0.6,line_color=clrs_bar[i]))
    fig2.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                       height=380,margin=dict(t=20,b=20),
                       legend=dict(orientation="h",y=-0.1))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Clustering Results":
    st.markdown("## 🔵 Clustering Results (Unsupervised)")
    st.markdown("KMeans on **112,120** images — true labels **never used for training**.")

    ca1,ca2 = st.columns(2)
    ca1.metric("🏆 Best k","k = 2")
    ca2.metric("📐 Silhouette Score","0.1529")

    st.markdown("---")
    cb1,cb2 = st.columns([1.3,1])
    with cb1:
        fig = px.line(df_sil,x="k",y="Silhouette",markers=True,
                      color_discrete_sequence=["#E74C3C"])
        fig.add_scatter(x=[2],y=[0.1529],mode="markers",name="Best k=2",
                        marker=dict(size=16,color="#E74C3C",symbol="star"))
        fig.update_layout(height=280,margin=dict(t=10,b=10),
                          xaxis=dict(tickvals=[2,3,4,5,6,8]))
        st.plotly_chart(fig, use_container_width=True)
    with cb2:
        df_s2 = df_sil.copy()
        df_s2["Best?"] = df_s2["k"].apply(lambda x:"⭐ Best" if x==2 else "")
        st.dataframe(df_s2, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Cluster × True Label Cross-Tabulation")
    df_ct = pd.DataFrame({
        "Cluster":["Cluster 0","Cluster 1","Total"],
        "Normal (0)":[69968,40721,110689],
        "Pneumonia (1)":[664,767,1431],
        "Total":[70632,41488,112120],
    })
    st.dataframe(df_ct, use_container_width=True, hide_index=True)

    fig3 = go.Figure(data=[
        go.Bar(name="Normal",x=["Cluster 0","Cluster 1"],
               y=[69968,40721],marker_color="#3498DB"),
        go.Bar(name="Pneumonia",x=["Cluster 0","Cluster 1"],
               y=[664,767],marker_color="#E74C3C"),
    ])
    fig3.update_layout(barmode="stack",height=300,
                       yaxis_title="Image Count",
                       legend=dict(orientation="h",y=-0.2),
                       margin=dict(t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.info("Pneumonia cases spread across both clusters — frozen DenseNet121 features "
            "capture general radiographic patterns rather than pneumonia-specific "
            "infiltrates. Fine-tuning would improve cluster-label alignment.")

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ PCA Visualisation":
    st.markdown("## 🖼️ PCA Cluster Visualisation")
    st.markdown("2-D PCA of all **112,120** DenseNet121 feature vectors coloured by "
                "KMeans cluster (k=2). Marker shape = true label *(analysis only)*.")

    if os.path.isfile(PNG):
        st.image(Image.open(PNG),
                 caption="Fig 1. KMeans clusters (k=2) on PCA-projected features. "
                         "○ = Normal, ✕ = Pneumonia.",
                 use_container_width=True)

    st.markdown("---")
    pc1,pc2 = st.columns(2)
    pc1.metric("PC1 Explained Variance","10.2%")
    pc2.metric("PC2 Explained Variance","6.1%")

    if os.path.isfile(PCA_PARQUET):
        st.markdown("### Interactive Scatter (sample)")
        with st.spinner("Loading..."):
            df_p = pd.read_parquet(PCA_PARQUET,
                                   columns=["pca_x","pca_y","prediction","label"])
            df_p["Cluster"]    = df_p["prediction"].map({0:"Cluster 0",1:"Cluster 1"})
            df_p["True Label"] = df_p["label"].map({0:"Normal",1:"Pneumonia"})
        fig4 = px.scatter(
            df_p,x="pca_x",y="pca_y",color="Cluster",symbol="True Label",
            color_discrete_map={"Cluster 0":"#3498DB","Cluster 1":"#E74C3C"},
            symbol_map={"Normal":"circle","Pneumonia":"x"},
            opacity=0.5,height=480,
            labels={"pca_x":"PC1","pca_y":"PC2"},
            title="Interactive PCA Scatter")
        fig4.update_traces(marker=dict(size=5))
        fig4.update_layout(margin=dict(t=40,b=10),legend_title_text="")
        st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Dataset & Pipeline":
    st.markdown("## 📁 Dataset & Pipeline Details")
    t1,t2 = st.tabs(["📦 Dataset","⚙️ Technology Stack"])
    with t1:
        da,db = st.columns(2)
        da.markdown("""
        | Attribute | Value |
        |---|---|
        | Dataset | NIH ChestX-ray14 |
        | Total images | **112,120** |
        | Unique patients | 30,805 |
        | Format | PNG (1024×1024) |
        | Labels | 14 diseases (multi-label) |
        | Train/Val | 86,524 |
        | Test | 25,596 |
        """)
        db.markdown("""
        | Attribute | Value |
        |---|---|
        | Pneumonia | 1,431 (1.3%) |
        | Normal | 110,689 (98.7%) |
        | Imbalance ratio | 77 : 1 |
        | Feature dim | 1,024-d |
        | Features size | 839 MB |
        | Archives | 12 × tar.gz (~42 GB) |
        """)
    with t2:
        st.markdown("""
        | Component | Technology | Version |
        |---|---|---|
        | Deep Features | PyTorch + DenseNet121 | 2.15.0 |
        | Storage | Apache Parquet | pyarrow 26.0.1 |
        | Processing | Apache Spark | PySpark 4.3.0 |
        | Classification | Spark MLlib LR/RF/GBT | — |
        | Clustering | Spark MLlib KMeans+PCA | — |
        | Dashboard | Streamlit + Plotly | — |
        | Java | OpenJDK | 25.0.2 |
        | Python | CPython | 3.15.2 |
        | OS | Windows 11 | — |
        """)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Download Report":
    st.markdown("## 📄 IEEE Project Report")
    if os.path.isfile(PDF):
        st.success("✅ IEEE-format report is ready!")
        with open(PDF,"rb") as f:
            st.download_button(
                label="📥 Download IEEE Report (PDF)",
                data=f,
                file_name="Pneumonia_Detection_MMDS_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.markdown("""
        ### Report Contents
        | Section | Content |
        |---|---|
        | Abstract | Pipeline overview + key results |
        | I. Introduction | Problem, contributions |
        | II. Related Work | CheXNet, NIH dataset, distributed ML |
        | III. Dataset | NIH stats, label strategy |
        | IV. Methodology | DenseNet121, classification, clustering |
        | V. Results | All tables + PCA figure |
        | VI. Discussion | Analysis + limitations |
        | VII. Conclusion | Summary + future work |
        | References | 10 IEEE citations |
        """)
    if os.path.isfile(PNG):
        st.image(PNG, caption="Fig 1. PCA Cluster Visualisation (embedded in report)",
                 width=500)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#95A5A6;font-size:0.8rem'>"
    "MMDS Project — Pneumonia Detection from Chest X-Rays using Apache Spark &amp; DenseNet121 "
    "| NIH ChestX-ray14 (112,120 images)"
    "</p>", unsafe_allow_html=True)
