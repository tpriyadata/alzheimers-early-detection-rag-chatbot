## 🚀 Live Demo
👉 [Try the Chatbot](https://alzheimers-early-detection-rag-chatbot-cdb5nsrc65ambm5cvwulug.streamlit.app/)

# alzheimers-early-detection-rag-chatbot

### Alzheimer’s Early Detection Research Chatbot

**Retrieval-Augmented Generation (RAG) + Large Language Models**

## Overview

The **Alzheimer’s Early Detection Research Chatbot** is an AI-powered research assistant designed to **retrieve, summarize, and answer questions** about Alzheimer’s disease using **trusted medical research sources**.

The system leverages **Retrieval-Augmented Generation (RAG)** to ground LLM responses in real scientific literature, reducing hallucinations and ensuring evidence-based outputs.

This project serves as a **portfolio-ready demonstration of end-to-end AI engineering**, combining data ingestion, vector search, embeddings, and LLM-based generation.

## Problem Statement

Alzheimer’s research is rapidly evolving, with thousands of publications across journals and clinical studies. Researchers, students, and clinicians face challenges such as:

* Time-consuming literature review
* Difficulty identifying early-detection biomarkers
* Risk of misinformation from generic AI tools

**Goal**:
Build a **grounded AI chatbot** that provides **accurate, research-backed answers** about Alzheimer’s early detection.

## 🌍 Real World Impact

780 million people worldwide are at risk 
of Alzheimer's by 2050 (WHO, 2023)

Early detection can slow progression 
by up to 40% with timely intervention

80% of rural communities have NO access 
to neurological specialists

This system gives ANY clinician — regardless 
of location or resources — access to 
evidence-based early detection support

## Key Capabilities

* **Retrieves relevant research papers** from trusted sources
*  **Summarizes complex medical literature**
*  **Answers natural-language questions** using retrieved evidence
*  **Reduces hallucinations** through RAG
*  **Cites source context** for transparency
*  **Interactive chatbot interface**

##  System Architecture
```
User Question
     ↓
Query Embedding
     ↓
Vector Database Search
     ↓
Relevant Research Chunks
     ↓
LLM with Retrieved Context (RAG)
     ↓
Evidence-Grounded Answer
```

##  Data Sources

Only **open and trusted medical sources** are used:

* PubMed (abstracts & open-access papers)
* NIH / NLM publications
* WHO Alzheimer’s reports
* ClinicalTrials.gov (metadata)

⚠️ No patient-level or private health data is used.

##  Tech Stack

* **Language:** Python
* **NLP / ML:**

  * Hugging Face Transformers
  * Sentence Transformers
* **RAG Framework:** LangChain / LlamaIndex
* **Vector Store:** FAISS / ChromaDB
* **LLMs:** Open-source or API-based LLMs
* **Primary LLM:** Claude (Anthropic API)
* **Fallback:** Mistral-7B (open source, offline use)
* **Embeddings:** all-MiniLM-L6-v2 (Sentence Transformers)
* **Backend / UI:** FastAPI / Streamlit
* **Evaluation:** Custom relevance & faithfulness checks

## Project Structure
```
alzheimer-rag-chatbot/
│
├── data/                 # Raw and processed research documents
├── ingestion/            # Data fetching & loading pipelines
├── preprocessing/        # Cleaning and chunking logic
├── embeddings/           # Embedding generation
├── vector_store/         # FAISS / Chroma setup
├── retrieval/            # Semantic search logic
├── generation/           # LLM prompts & response generation
├── evaluation/           # Faithfulness & relevance checks
├── api/                  # Chatbot API / UI logic
├── configs/              # Configuration files
├── tests/                # Unit tests
└── README.md
```
##  Example Questions

* *What are early biomarkers for Alzheimer’s disease?*
* *How is PET imaging used in early diagnosis?*
* *What blood-based biomarkers are currently studied?*
* *What is Mild Cognitive Impairment (MCI)?*

## Evaluation Focus

* Retrieval relevance
* Answer faithfulness to source documents
* Hallucination reduction
* Context coverage
* User interpretability

## Ethics & Safety

* Research and educational use only
* No diagnosis or medical advice
* No patient-identifiable data
* Clear AI-generated content disclaimer
* Human-in-the-loop decision making
  
## Future Enhancements

* Knowledge graph for biomarkers & studies
* Multilingual research support
* Temporal trend analysis of publications
* Feedback-based retrieval tuning
* Integration with imaging metadata (MRI/PET)

## Why This Project Stands Out

✔ Demonstrates **RAG architecture mastery**
✔ Shows **healthcare-aware AI design**
✔ Emphasizes **accuracy & trust**
✔ Built with **production-ready structure**
✔ Ideal for **GenAI / NLP / AI Engineer roles**

## Author

**TamilPriya Balaji**




