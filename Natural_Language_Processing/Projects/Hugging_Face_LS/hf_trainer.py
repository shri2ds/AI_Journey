import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.optim import AdamW
from NLP.Projects.Log_Sentinel.log_generator import LogDatasetGenerator

class LogHFDataset(Dataset):
    def __init__(self, size, tokenizer, max_len=16):
        self.generator = LogDatasetGenerator()
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Generate raw strings instead of IDs for the HF Tokenizer
        self.logs = []
        self.labels = []
        for _ in range(size):
            is_attack = torch.rand(1).item() < 0.2
            path, label = self.generator.generate_session(is_attack)
            log_str = " ".join([self.generator.id_to_event[int(i)] for i in path])
            self.logs.append(log_str)
            self.labels.append(label)

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            self.logs[item],
            padding = 'max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

# Setup Model & Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

for param in model.distilbert.parameters():
    param.requires_grad = False

# DataLoaders
train_dataset = LogHFDataset(1000, tokenizer, max_len=16)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# The Fine-Tuning Loop
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

print("--- Training HuggingFace Sentinel ---")
for epoch in range(3): # HF models converge fast!
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Complete. Loss: {loss.item():.4f}")

#    Testing on the sample log
def test_inference(sample_log):
    inputs = tokenizer(sample_log, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    print(f"Log: {sample_log} | Result: {'ATTACK' if pred == 1 else 'NORMAL'}")

test_inference("LOGIN SUDO DELETE_LOGS LOGOUT")

#   Defining the path 
save_directory = "./saved_log_sentinel"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

#   Save model & it's config
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"✅ Model and Tokenizer successfully saved to {save_directory}")
