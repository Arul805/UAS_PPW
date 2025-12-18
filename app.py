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



def pagerank_manual(G, alpha=0.85, max_iter=100, tol=1e-6):
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return {}

    node_index = {node: i for i, node in enumerate(nodes)}

    # Adjacency matrix berbobot
    A = np.zeros((n, n))

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        i, j = node_index[u], node_index[v]
        A[j][i] += w
        A[i][j] += w  # karena graph tidak berarah

    # Normalisasi kolom
    column_sums = A.sum(axis=0)
    column_sums[column_sums == 0] = 1
    M = A / column_sums

    # PageRank init
    pr = np.ones(n) / n

    for _ in range(max_iter):
        new_pr = alpha * M @ pr + (1 - alpha) / n
        if np.linalg.norm(new_pr - pr, 1) < tol:
            break
        pr = new_pr

    return {nodes[i]: pr[i] for i in range(n)}

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

st.subheader("📝 Atau tempel teks dokumen")
manual_text = st.text_area(
    "Paste isi jurnal di sini",
    height=250
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
                
elif manual_text.strip() != "":
    text = manual_text

    st.subheader("📄 Contoh isi dokumen")
    st.text(text[:500])

    if st.button("🚀 Extract Keyword"):
        # Preprocessing
        tokens = preprocess_text(text)

        # Bangun graph
        G = build_word_graph(tokens)

        # PageRank
        pagerank = pagerank_manual(G)


        # Ambil top keyword
        pr_df = pd.DataFrame(pagerank.items(), columns=["Keyword", "PageRank"])
        pr_df = pr_df.sort_values(by="PageRank", ascending=False).head(top_k)

        st.subheader("📊 Top Keyword berdasarkan PageRank")
        st.dataframe(pr_df)

        # =========================
        # VISUALISASI GRAPH
        # =========================
        st.subheader("🕸️ Visualisasi Graph Kata")

        sub_nodes = pr_df["Keyword"].tolist()
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
