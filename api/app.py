# app.py - Alzheimer's Research Chatbot UI
# Run with: streamlit run api/app.py

import streamlit as st
import numpy as np
import json
import faiss
import os
import re
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Page Configuration ──────────────────
st.set_page_config(
    page_title="Alzheimer's Early Detection Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ── Hero Section ─────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px 0;">
    <h1>🧠 Alzheimer's Early Detection Assistant</h1>
    <p style="font-size:18px; color:#4F4F4F;">
        Powered by PubMed Research + Claude AI
    </p>
    <p style="font-size:15px; color:#6E6E6E;">
        Ask evidence-based questions about early 
        detection, biomarkers, symptoms and research.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Medical Disclaimer ───────────────────
st.markdown("""
<div style="
    border-radius:14px;
    padding:20px;
    background-color:#FFF8E6;
    border-left:6px solid #E67E22;
    margin-bottom:25px;
">
    <h4>⚠️ Medical Disclaimer</h4>
    <p>This AI assistant is for 
    <strong>educational and research purposes only.</strong></p>
    <ul>
        <li>❌ Not a clinical diagnostic tool</li>
        <li>❌ Not a substitute for medical advice</li>
        <li>✅ Informational responses from research papers</li>
        <li>✅ Always consult a licensed healthcare professional</li>
    </ul>
    <p style="font-size:13px; color:#555;">
    For medical emergencies call 911 immediately.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Build Pipeline If Needed ─────────────
def build_pipeline_if_needed():
    """Auto-build FAISS index on first run"""

    if not os.path.exists(
        "vector_store/faiss_index/alzheimer.index"
    ):
        st.info("⚙️ Building research database...")

        from Bio import Entrez
        Entrez.email = "research@alzheimer.com"

        handle = Entrez.esearch(
            db="pubmed",
            term="Alzheimer's early detection biomarkers",
            retmax=10
        )
        results = Entrez.read(handle)
        handle.close()

        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=results["IdList"],
            rettype="abstract",
            retmode="xml"
        )
        papers = Entrez.read(fetch_handle)
        fetch_handle.close()

        chunks = []
        for article in papers["PubmedArticle"]:
            title = str(
                article["MedlineCitation"]["Article"]["ArticleTitle"]
            )
            try:
                abstract = str(
                    article["MedlineCitation"]["Article"]
                    ["Abstract"]["AbstractText"][0]
                )
            except KeyError:
                abstract = "No abstract available"

            text = re.sub(
                r'\s+', ' ', title + ". " + abstract
            ).strip()
            chunks.append({
                "pmid": str(article["MedlineCitation"]["PMID"]),
                "title": title,
                "text": text,
                "source": "PubMed"
            })

        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/chunks.json", "w") as f:
            json.dump(chunks, f)

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.encode(texts)
        embeddings = embeddings.astype(np.float32)

        os.makedirs("vector_store/faiss_index", exist_ok=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(
            index,
            "vector_store/faiss_index/alzheimer.index"
        )

        st.success(" Research database built!")
        st.rerun()

# ── Load Components ──────────────────────
@st.cache_resource
def load_components():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(
        "vector_store/faiss_index/alzheimer.index"
    )

    with open("data/processed/chunks.json", "r") as f:
        chunks = json.load(f)

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    return model, index, chunks, client

# ── Initialize ───────────────────────────
build_pipeline_if_needed()
model, index, chunks, client = load_components()

st.success(" Research database loaded!")
st.divider()

# ── Retrieval Function ───────────────────
def retrieve_chunks(question, top_k=3):

    question_vector = model.encode([question])
    question_vector = question_vector.astype(np.float32)
    distances, indices = index.search(question_vector, k=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "title": chunks[idx]["title"],
            "pmid": chunks[idx]["pmid"],
            "distance": float(dist)
        })
    return results

# ── Generation Function ──────────────────
def ask_chatbot(question):

    relevant_chunks = retrieve_chunks(question)

    context = ""
    for i, chunk in enumerate(relevant_chunks):
        context += f"Source {i+1}: {chunk['title']}\n"
        context += f"{chunk['text']}\n\n"

    prompt = f"""You are an Alzheimer's disease research assistant.
Answer using ONLY the provided research context.
Always cite which source you used.
If the context doesn't contain the answer, say so honestly.

RESEARCH CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.content[0].text,
        "sources": relevant_chunks
    }

# ── Chat History ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ───────────────────────────
if prompt := st.chat_input(
    "💬 Ask about Alzheimer's research..."
):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching research papers..."):
            result = ask_chatbot(prompt)

        st.markdown(result["answer"])

        st.divider()
        st.markdown("** Sources Used:**")
        for i, source in enumerate(result["sources"]):
            st.markdown(
                f"{i+1}. *{source['title'][:80]}...*"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })

# ── Footer ───────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:13px;">
    Built by <strong>Priya</strong> | 
    Data Analyst & NLP Engineer |
    Powered by PubMed + Claude AI
</div>
""", unsafe_allow_html=True)
