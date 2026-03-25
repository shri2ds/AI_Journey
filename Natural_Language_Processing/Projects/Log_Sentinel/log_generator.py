import torch
import random

class LogDatasetGenerator:
    def __init__(self):
        # Our "Vocabulary" of system events
        self.events = ["PAD", "LOGIN", "VIEW", "UPLOAD", "DOWNLOAD", "SUDO", "EDIT_CONFIG", "DELETE_LOGS", "LOGOUT"]
        self.event_to_id = {event: i for i, event in enumerate(self.events)}
        self.id_to_event = {i: event for event, i in self.event_to_id.items()}

    def generate_session(self, is_attack=False):
        if not is_attack:
            # Normal Flow: Standard user behavior
            path = ["LOGIN", random.choice(["VIEW", "UPLOAD", "DOWNLOAD"]), "LOGOUT"]
        else:
            # Attack Flow: Malicious escalation
            path = ["LOGIN", "SUDO", random.choice(["EDIT_CONFIG", "DELETE_LOGS"]), "LOGOUT"]

        # Convert to IDs and Pad to a fixed length (e.g., 5)
        ids = [self.event_to_id[e] for e in path]
        while len(ids) < 5:
            ids.append(self.event_to_id["PAD"])

        return torch.tensor(ids), torch.tensor(1 if is_attack else 0)

    def create_batch(self, size=1000):
        data, labels = [], []
        for _ in range(size):
            is_attack = random.random() < 0.2  # 20% attacks
            x, y = self.generate_session(is_attack)
            data.append(x)
            labels.append(y)

        return torch.stack(data), torch.stack(labels)


if __name__ == "__main__":
    generator = LogDatasetGenerator()
    x_train, y_train = generator.create_batch(size=10)

    print("--- Sample Generated Logs (Tokens) ---")
    print(x_train[0])
    print(f"Label (0=Normal, 1=Attack): {y_train[0]}")

    # Human readable check
    sample_events = [generator.id_to_event[int(i)] for i in x_train[0]]
    print(f"Human Readable: {' -> '.join(sample_events)}")
