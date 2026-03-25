import spacy
import nltk
from nltk.stem import PorterStemmer

# Setup NLTK (The Stemmer)
stemmer = PorterStemmer()

# Setup Spacy (The Lemmatizer)
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# The Tricky List
words = ["running", "flies", "better", "meeting", "universities", "mice"]

print(f"{'Original':<15} | {'Stemmer (NLTK)':<15} | {'Lemmatizer (Spacy)':<15}")
print("-" * 50)

for word in words:
    # Stemming (Rule-based chopping)
    stem_result = stemmer.stem(word)
    
    # Lemmatization (Context-based lookup)
    # Spacy requires a document context, so we process the word
    doc = nlp(word)
    lemma_result = doc[0].lemma_
    
    print(f"{word:<15} | {stem_result:<15} | {lemma_result:<15}")
