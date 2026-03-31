from transformers import DistilBertForSequenceClassification, DistilBertConfig
from torch.optim import AdamW

# Load the Pre-trained "Brain"
model_name = "distilbert-base-uncased"
config = DistilBertConfig.from_pretrained(model_name, num_labels=2)
hf_model = DistilBertForSequenceClassification.from_pretrained(model_name, config=config)

# Production Reality Check: Freezing the Backbone
for param in hf_model.distilbert.parameters():
    param.requires_grad = False

# The Optimizer
optimizer = AdamW(hf_model.parameters(), lr=5e-5)

print(f"Total Parameters: {sum(p.numel() for p in hf_model.parameters())}")
print(f"Trainable Parameters: {sum(p.numel() for p in hf_model.parameters() if p.requires_grad)}")
