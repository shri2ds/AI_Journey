# 🗣️ Natural Language Processing

> **Focus:** Transitioning from "Pixels" (Computer Vision) to "Sequences" (Text). Mastering the pipeline from basic Regex to Large Language Models (LLMs).

This directory contains code, notes, and projects for **Month 3** of the AI Engineering Journey. The goal is to build a deep understanding of how machines understand human language, culminating in building a **RAG (Retrieval-Augmented Generation)** system.

---

## 📅 Accelerated Curriculum (20 Days)

We are condensing a traditional 30-day curriculum into an intensive 20-day sprint to catch up on the timeline.

### **Week 1: The Words (Foundations)**
*Processing raw text into mathematical vectors.*
* **Day 1:** Text Preprocessing (Regex, Normalization). 
* **Day 2:** Tokenization (Spacy, NLTK) & Lemmatization.
* **Day 3:** Vectorization (Bag of Words, TF-IDF).
* **Day 4:** Word Embeddings (Word2Vec, GloVe).
* **Day 5:** **Project:** 📄 Resume Keyword Extractor.

### **Week 2: The Sequence (RNNs)**
*Understanding memory and context in sequential data.*
* **Day 6:** Recurrent Neural Networks (RNNs) - The loop. 
* **Day 7:** LSTMs & GRUs (Solving Vanishing Gradients).
* **Day 8:** Seq2Seq Models (Encoder-Decoder Architecture).
* **Day 9:** **Project:** 🔮 Next Word Predictor (Shakespeare Bot).

### **Week 3: The Revolution (Transformers)**
*The "Attention Is All You Need" era.*
* **Day 10:** The Attention Mechanism (Bahdanau).
* **Day 11:** The Transformer Architecture (Self-Attention, Multi-Head). 
* **Day 12:** BERT (Bidirectional Encoder Representations).
* **Day 13:** GPT (Generative Pre-trained Transformer).
* **Day 14:** **Project:** 🎭 Sentiment Analysis with BERT.

### **Week 4: The LLM Era (Modern NLP)**
*Productionizing Large Language Models.*
* **Day 15:** HuggingFace `transformers` Library.
* **Day 16:** Fine-Tuning Techniques (LoRA, PEFT).
* **Day 17:** RAG (Retrieval Augmented Generation) Concepts.
* **Day 18:** Vector Databases (ChromaDB / Pinecone).
* **Day 19-20:** **Capstone Project:** 🤖 Chat with PDF (RAG Bot).

---

## 🛠️ Tech Stack

* **Libraries:** `spaCy`, `NLTK`, `scikit-learn`, `PyTorch`
* **Deep Learning:** `transformers` (HuggingFace), `torch.nn`
* **Vector Stores:** `ChromaDB` or `FAISS`
* **Environment:** Python 3.10+

---

## 🚀 Setup

1.  **Install Core NLP Libraries:**
    ```bash
    pip install spacy nltk scikit-learn
    python -m spacy download en_core_web_sm
    ```

2.  **Install Deep Learning Utilities (Week 2+):**
    ```bash
    pip install torch transformers sentence-transformers
    ```

---

## 📂 Directory Structure

```text
Natural_Language_Processing/
├── Preprocessing/    # Regex, Cleaning, Spacy
├── Tokenization/     # Stemming vs Lemmatization
├── ...
└── Projects/
    ├── Resume_Parser/
    └── RAG_Chatbot/
