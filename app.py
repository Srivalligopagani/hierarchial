import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ------------------------------------------------
# CSS
# ------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f4f7fc;
}

.hero{
    padding:25px;
    border-radius:15px;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.metric-box{
    background:white;
    padding:15px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HERO
# ------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🌳 Hierarchical Clustering Dashboard</h1>
<p>Iris Dataset Analysis using Agglomerative Clustering</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("⚙️ Parameters")

clusters_num = st.sidebar.slider(
    "Number of Clusters",
    2,
    6,
    3
)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# ------------------------------------------------
# SCALING
# ------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df)

# ------------------------------------------------
# MODEL
# ------------------------------------------------

model = AgglomerativeClustering(
    n_clusters=clusters_num,
    linkage=linkage_method
)

cluster_labels = model.fit_predict(X_scaled)

df["Cluster"] = cluster_labels

# ------------------------------------------------
# METRICS
# ------------------------------------------------

st.subheader("📊 Dataset Overview")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Total Samples", len(df))

with c2:
    st.metric("Features", len(iris.feature_names))

with c3:
    st.metric("Clusters", clusters_num)

# ------------------------------------------------
# DATASET
# ------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head(15), use_container_width=True)

# ------------------------------------------------
# DENDROGRAM
# ------------------------------------------------

st.subheader("🌲 Dendrogram")

linked = linkage(
    X_scaled,
    method=linkage_method
)

fig, ax = plt.subplots(figsize=(12,5))

dendrogram(
    linked,
    ax=ax,
    leaf_rotation=90
)

ax.set_title("Hierarchical Clustering Dendrogram")

st.pyplot(fig)

# ------------------------------------------------
# CLUSTER DISTRIBUTION
# ------------------------------------------------

st.subheader("📈 Cluster Distribution")

cluster_count = (
    pd.Series(cluster_labels)
    .value_counts()
    .sort_index()
)

fig2, ax2 = plt.subplots()

ax2.bar(
    cluster_count.index.astype(str),
    cluster_count.values
)

ax2.set_xlabel("Cluster")
ax2.set_ylabel("Count")

st.pyplot(fig2)

# ------------------------------------------------
# SCATTER PLOT
# ------------------------------------------------

st.subheader("🎯 Cluster Visualization")

x_feature = st.selectbox(
    "Select X Feature",
    iris.feature_names,
    index=0
)

y_feature = st.selectbox(
    "Select Y Feature",
    iris.feature_names,
    index=2
)

fig3, ax3 = plt.subplots(figsize=(8,5))

scatter = ax3.scatter(
    df[x_feature],
    df[y_feature],
    c=cluster_labels
)

ax3.set_xlabel(x_feature)
ax3.set_ylabel(y_feature)
ax3.set_title("Cluster Visualization")

st.pyplot(fig3)

# ------------------------------------------------
# CLUSTER STATS
# ------------------------------------------------

st.subheader("📑 Cluster Statistics")

cluster_stats = (
    df.groupby("Cluster")
    .mean()
)

st.dataframe(
    cluster_stats,
    use_container_width=True
)

# ------------------------------------------------
# DOWNLOAD
# ------------------------------------------------

st.subheader("⬇️ Download Results")

csv = df.to_csv(index=False)

st.download_button(
    "Download CSV",
    csv,
    "hierarchical_clustering.csv",
    "text/csv"
)

# ------------------------------------------------
# THEORY
# ------------------------------------------------

with st.expander("📚 Learn About Hierarchical Clustering"):

    st.markdown("""
### What is Hierarchical Clustering?

Hierarchical Clustering is an unsupervised machine learning algorithm.

It creates a hierarchy of clusters using:

- Agglomerative (Bottom-Up)
- Divisive (Top-Down)

### Applications

- Customer Segmentation
- Market Basket Analysis
- Image Segmentation
- Biological Data Analysis

### Advantages

- No labels required
- Easy interpretation through dendrograms
- Works well for smaller datasets

### Disadvantages

- Computationally expensive
- Sensitive to noise
- Difficult on very large datasets
""")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.caption(
    "Machine Learning Mini Project | Hierarchical Clustering using Iris Dataset"
)