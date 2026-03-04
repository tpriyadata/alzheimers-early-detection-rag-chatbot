
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