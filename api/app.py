# app.py - Alzheimer's Research Chatbot UI
# Run with: streamlit run api/app.py

import streamlit as st
import numpy as np
import json
import faiss
import os
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# ── Page Configuration ──────────────────
st.set_page_config(
    page_title="Alzheimer's Research Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ── Title & Description ──────────────────
st.title("🧠 Alzheimer's Early Detection Chatbot")
st.markdown("""
*Powered by PubMed research + Claude AI*
Ask any question about Alzheimer's early detection.
""")

st.divider()
# Load the Model and Data
# ── Load Models & Data ───────────────────
# @st.cache_resource means:
# load once, reuse every time
# don't reload on every question!

def build_pipeline_if_needed():
    """Build FAISS index if it doesn't exist"""
    
    if not os.path.exists("vector_store/faiss_index/alzheimer.index"):
        
        st.info("⚙️ Building research database for first time...")
        
        # Fetch papers
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
        
        # Process papers
        import re
        chunks = []
        for article in papers["PubmedArticle"]:
            title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
            try:
                abstract = str(article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0])
            except KeyError:
                abstract = "No abstract available"
            
            text = re.sub(r'\s+', ' ', title + ". " + abstract).strip()
            chunks.append({
                "pmid": str(article["MedlineCitation"]["PMID"]),
                "title": title,
                "text": text,
                "source": "PubMed"
            })
        
        # Save chunks
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/chunks.json", "w") as f:
            json.dump(chunks, f)
        
        # Generate embeddings
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c["text"] for c in chunks]
        embeddings = embedding_model.encode(texts)
        embeddings = embeddings.astype(np.float32)
        
        # Build FAISS index
        os.makedirs("vector_store/faiss_index", exist_ok=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(
            index,
            "vector_store/faiss_index/alzheimer.index"
        )
        
        st.success("✅ Research database built!")
        st.rerun()


@st.cache_resource
def load_components():
    
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load FAISS index
    index = faiss.read_index(
        "vector_store/faiss_index/alzheimer.index"
    )
    
    # Load chunks
    with open("data/processed/chunks.json", "r") as f:
        chunks = json.load(f)
    
    # Load Claude client
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    return model, index, chunks, client

# Build pipeline if needed
build_pipeline_if_needed()

# Load everything
model, index, chunks, client = load_components()


st.success("✅ Research database loaded!")

#Key concept — @st.cache_resource:

# Without it:
# Every time user asks a question
# → reloads model (30 seconds!) ❌

# With it:
# Loads once when app starts
# → reuses same model forever ✅
# This is called "caching"

# ── Retrieval Function ───────────────────
def retrieve_chunks(question, top_k=3):
    
    # Convert question to vector
    question_vector = model.encode([question])
    question_vector = question_vector.astype(np.float32)
    
    # Search FAISS
    distances, indices = index.search(
        question_vector, k=top_k
    )
    
    # Collect results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "title": chunks[idx]["title"],
            "pmid": chunks[idx]["pmid"],
            "distance": float(dist)
        })
    
    return results

# ──Generation Functions ──────────────────
def ask_chatbot(question):
    
    # Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(question)
    
    # Build context
    context = ""
    for i, chunk in enumerate(relevant_chunks):
        context += f"Source {i+1}: {chunk['title']}\n"
        context += f"{chunk['text']}\n\n"
    
    # Build prompt
    prompt = f"""You are an Alzheimer's disease 
research assistant. Answer using ONLY the 
provided research context. Always cite sources.
If context doesn't contain the answer, say so.

RESEARCH CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    # Ask Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "answer": response.content[0].text,
        "sources": relevant_chunks
    }
# ── Chat History ─────────────────────────
# st.session_state keeps messages between questions

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ───────────────────────────
if prompt := st.chat_input("Ask about Alzheimer's research..."):
    
    # Show user question
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Show thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching research papers..."):
            result = ask_chatbot(prompt)
        
        # Show answer
        st.markdown(result["answer"])
        
        # Show sources
        st.divider()
        st.markdown("**📚 Sources Used:**")
        for i, source in enumerate(result["sources"]):
            st.markdown(
                f"{i+1}. *{source['title'][:80]}...*"
            )
    
    # Add answer to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })