import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from NLP.Projects.Log_Sentinel.log_generator import LogDatasetGenerator
from NLP.Projects.Log_Sentinel.integrated_transformer import LogSentinel
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Setup Data & Hyperparameters
generator = LogDatasetGenerator()
X_train, y_train = generator.create_batch(size=2000)
X_test, y_test = generator.create_batch(size=500)

VOCAB_SIZE = len(generator.events)
EMBED_DIM = 32
NUM_HEADS = 4
FF_DIM = 128
EPOCHS = 50
LR = 0.001

# 2. Initialize Model, Loss, and Optimizer
model = LogSentinel(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM)
criterion = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=LR)

# 3. Training Loop
print("--- Starting Training ---")
for epoch in range(EPOCHS):
    model.train()
    optimiser.zero_grad()

    # Forward Pass
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train.float())

    # Backward Pass
    loss.backward()
    optimiser.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

# 4. Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test).squeeze()
    predictions = (test_outputs > 0.5).float()
    y_true = y_test.numpy()

    # 1. Generate the Confusion Matrix
    cm = confusion_matrix(y_true, predictions)

    # 2. Detailed Report (Precision, Recall, F1-Score)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, predictions, target_names=['Normal', 'Attack']))

    # 3. Visual Plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Normal', 'Pred Attack'],
                yticklabels=['Actual Normal', 'Actual Attack'])
    plt.title("Log Sentinel: Confusion Matrix")
    plt.savefig('LogSentinel_ConfusionMatrix.png')
    plt.show()

    
    accuracy = (predictions == y_test).float().mean()
    print(f"\n✅ Final Test Accuracy: {accuracy.item()*100:.2f}%")


