import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import re
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import nltk

from PyPDF2 import PdfReader

# Download stopwords (untuk HuggingFace aman)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# =========================
# PREPROCESSING
# =========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)   # hapus simbol & angka
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

# =========================
# BUILD WORD GRAPH
# =========================
def build_word_graph(tokens):
    G = nx.Graph()

    # Unigram
    for word in tokens:
        if not G.has_node(word):
            G.add_node(word)

    # Bigram (hubungan kata)
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if G.has_edge(w1, w2):
            G[w1][w2]['weight'] += 1
        else:
            G.add_edge(w1, w2, weight=1)

    return G





# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Keyword Extraction Graph", layout="wide")

st.title("🔑 Keyword Extraction berbasis Graph & PageRank")
st.write("Upload dokumen teks → bangun graph kata → hitung PageRank → ambil keyword penting")

uploaded_file = st.file_uploader(
    "📄 Upload dokumen (.txt / .pdf)",
    type=["txt", "pdf"]
)



top_k = st.slider("Jumlah keyword yang ditampilkan", 5, 30, 20)

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                    text += page.extract_text()
                

    st.subheader("📄 Contoh isi dokumen")
    st.text(text[:500])

    if st.button("🚀 Extract Keyword"):
        # Preprocessing
        tokens = preprocess_text(text)

        # Bangun graph
        G = build_word_graph(tokens)

        # =========================
        # CENTRALITY MEASURES
        # =========================
        pagerank = nx.pagerank(G, weight="weight")
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
        closeness_centrality = nx.closeness_centrality(G)

        # =========================
        # DATAFRAME CENTRALITY
        # =========================
        centrality_df = pd.DataFrame({
            "Keyword": list(pagerank.keys()),
            "PageRank": list(pagerank.values()),
            "Degree": [degree_centrality[k] for k in pagerank.keys()],
            "Betweenness": [betweenness_centrality[k] for k in pagerank.keys()],
            "Closeness": [closeness_centrality[k] for k in pagerank.keys()]
        })

        centrality_df = centrality_df.sort_values(
            by="PageRank", ascending=False
        ).head(top_k)

        st.subheader("📊 Top Keyword & Centrality")
        st.dataframe(centrality_df)


        # =========================
        # VISUALISASI GRAPH
        # =========================
        st.subheader("🕸️ Visualisasi Graph Kata")

        sub_nodes = centrality_df["Keyword"].tolist()
        subgraph = G.subgraph(sub_nodes)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph, seed=42)
        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_size=800,
            font_size=10,
            edge_color="gray"
        )
        st.pyplot(plt)
