"""
Project: Agentic LogSentinel - Breach Analysis Module
Module: Model Taxonomy & Pipeline Inference

PURPOSE:
This module evaluates different Transformer architectures for automated security incident response. In a production Agentic workflow, selecting the right architectural "brain" is critical for balancing latency, cost, and accuracy.

TRANSFORMER TAXONOMY OBSERVED:
1. ENCODER-ONLY (e.g., BERT, DistilBERT): 
   - Bi-directional context. 
   - Best for: Log Classification, Named Entity Recognition (NER).
   - Limitation: Cannot generate fluid natural language responses.

2. DECODER-ONLY (e.g., GPT-2/3/4, Llama, DistilGPT2): 
   - Auto-regressive (predicts next token).
   - Best for: Reasoning Agents, zero-shot summarization, and creative instruction following.

3. ENCODER-DECODER (e.g., T5, BART): 
   - Seq-to-Seq architecture.
   - Best for: Complex summarization and translation where input understanding must be mapped to a completely new output structure.

EXPERIMENTAL GOAL:
Observing the 'hallucination' and 'repetition' tendencies of small-scale Decoder models (82M params) vs. the high-fidelity reasoning of Large Language Models (LLMs).
"""

from transformers import pipeline

summarizer = pipeline("text-generation", model="distilgpt2")

breach_context = "ALERT: Unauthorized SUDO access detected for user 'admin_01'. Sequence: LOGIN -> SUDO -> DELETE_LOGS."

prompt = f"Summarize this security breach and suggest a fix: {breach_context}\nSummary:"

result = summarizer(prompt, max_new_tokens=50, truncation=True)
print(f"--- Security Agent Output ---\n{result[0]['generated_text']}")
