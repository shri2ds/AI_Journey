# 🛡️ LogSentinel: Transformer-Based Anomaly Detection for System Telemetry

### **The Problem**
Traditional rule-based security systems often miss complex, multi-hop "low-and-slow" attacks in system telemetry. As an **Applied Scientist** transition from a **Platform Engineering** background, I built LogSentinel to leverage deep learning for sequence-based anomaly detection.

### **The Solution**
LogSentinel is an Encoder-only Transformer model built entirely **from scratch in PyTorch**. It processes sequences of system events (represented as "tokens") and classifies them as "Normal" or a "Security Attack" based on the learned context of the user session.

### **Architecture**
The pipeline integrates advanced modular components designed for production-level scalability:
1.  **Custom Vocab & Embedding:** System events are mapped to dense 32-dimensional vectors 
2.  **Positional Encoding (Sinusoidal):** Injects temporal order into the sequence, allowing the model to understand the context of event chronology (e.g., `LOGIN -> SUDO` vs. `SUDO -> LOGIN`).
3.  **Transformer Blocks:** Feature two multi-head attention layers with residual connections and layer normalization for stable gradient flow.
4.  **Classification Head:** Global average pooling and a sigmoid activation layer to output an attack probability.

### **Data Simulation**
To make this realistic for a Cybersecurity use-case, I generated synthetic data simulating user behavior:
* **Normal:** Logical flows like `LOGIN -> VIEW -> UPLOAD -> LOGOUT`.
* **Attack:** Malicious escalations such as `LOGIN -> SUDO -> EDIT_CONFIG -> LOGOUT`.

### **🛡️ Model Evaluation & Analysis**

We utilize a **Confusion Matrix** for a "Production Reality Check." While raw accuracy is easy to achieve on balanced synthetic data, we must verify that the model is minimizing both **False Positives (FP)** and **False Negatives (FN)**.

![LogSentinel ConfusionMatrix](./Log_Sentinel/LogSentinel_ConfusionMatrix.png)

**Analysis of Perfect Recall/Precision:**
The matrix demonstrates that the 2-block Transformer perfectly differentiated between the Normal and Attack sequences.

In a real-world deployment, this usually signals over-fitting to the synthetic data's simplicity. In production, we would move to a messy, imbalanced dataset and analyze these trade-offs:

* **Handling False Negatives (FN):** Missing an attack is catastrophic. Our strategy: prioritize Recall by lowering the Sigmoid threshold (e.g., from 0.5 to 0.3) to be more aggressive in threat detection.
* **Handling False Positives (FP):** Causing "alert fatigue" can be problematic. Our strategy: use weighted loss to penalize the model heavily for misidentifying attacks, or increase the threshold for high-fidelity alerts.

### **Project Structure**
The project is modular and referenced my own "Transformer Core" library:

```text
Log_Sentinel
├── log_generator.py          # Generates synthetic telemetry logs
├── integrated_transformer.py  # Assembles core transformer blocks
└── trainer.py                 # Runs training loop with BCELoss and sklearn evaluation
```

### 🚀 Execution Guide

Follow these steps to generate the data and train the model locally.

#### 1. Environment Setup
Ensure you have PyTorch and Scikit-Learn installed:
```bash
pip install torch matplotlib seaborn scikit-learn
```

#### 2. Project Initialization
Ensure your directory structure is organized as follows to allow for modular imports:
```text
NLP/
├── Transformer_Core/           # Core Transformer blocks
└── Log_Sentinel/            # Project files
```

#### 3. Running the Trainer
To run the training and evaluation pipeline, execute the trainer from the root NLP directory to resolve module paths:
```bash
# Navigate to your NLP root
cd /path/to/your/NLP/

# Run as a module
python -m Log_Sentinel.trainer
```

#### 4. Expected Output
The script will:

* Generate 2,000 synthetic log sequences.
* Train the 2-layer Transformer for 50 epochs.
* Display a Confusion Matrix and a Classification Report showing Precision, Recall, and F1-Score & Accuracy.
