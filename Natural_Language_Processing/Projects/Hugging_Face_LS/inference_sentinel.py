import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the Persisted Brain
model_path = "./saved_log_sentinel"   # Make sure the model is present in the current directory by building through hf_trainer.py
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

model.eval()

def predict_log(raw_log_string):
    # Preprocess the live input
    inputs = tokenizer(
        raw_log_string,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=16
    )

    # Forward Pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply Softmax to get probabilities (0 to 1)
        probs = torch.nn.functional.softmax(logits, dim=1)
        attack_prob = probs[0][1].item()

    status = "🚨 ATTACK DETECTED" if attack_prob > 0.5 else "NORMAL"

    print(f"Log: {raw_log_string}")
    print(f"Attack Probability: {attack_prob:.4f} | Status: {status}\n")


if __name__ == "__main__":
    # Test cases for your live sensor
    predict_log("LOGIN VIEW_DASHBOARD LOGOUT")
    predict_log("LOGIN SUDO DELETE_LOGS LOGOUT")
    predict_log("LOGIN EDIT_CONFIG DELETE_LOGS")

