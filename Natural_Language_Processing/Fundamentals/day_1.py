import re
import spacy

# Load English model (The "Brain")
nlp = spacy.load("en_core_web_sm")

raw_text = "The AI/ML Engineer job is GREAT! Contact us at hire-me@google.com. 📱 #AI"

def 'n'(text):
    # Normalization (Lowercasing)
    text = text.lower()

    # Regex
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Lemmatization & Tokenization
    doc = nlp(text)

    clean_tokens = []
    for token in doc:
        # Remove Stop Words and Whitespace
        if not token.is_stop and not token.is_space:
            # Append the lemma (root form)
            clean_tokens.append(token.lemma_)

    return clean_tokens

# Run it
processed = clean_text(raw_text)
print(f"Original: {raw_text}")
print(f"Processed: {processed}")
