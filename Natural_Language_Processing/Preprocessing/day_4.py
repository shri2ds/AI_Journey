from gensim.models import Word2Vec
import gensim.downloader as api

sentences = [
    ["i", "love", "machine", "learning"],
    ["i", "love", "artificial", "intelligence"],
    ["i", "enjoy", "deep", "learning"],
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["the", "king", "is", "man"],
    ["the", "queen", "is", "woman"]
]

# Train the model
model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, epochs=100)

# Similarity
print("--- Similarity Scores ---")
similarity_score = model.wv.similarity("machine", "artificial")
print(f"Similarity (Machine vs Artificial): {similarity_score:.4f}")

print(f"Similarity (King vs Queen): {model.wv.similarity('king', 'queen'):.4f}")

# Synonyms
print("\n--- Most Similar to 'Learning' ---")
similar_words = model.wv.most_similar("learning", topn=3)
for word, score in similar_words:
    print(f"{word}: {score:.4f}")

# The Math (King - Man + Woman)
result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)

# Leveraging a pre-trained model on Twitter (25 dim vectors, small but decent)
print("Downloading pre-trained model...")
model_wiki = api.load("glove-wiki-gigaword-100")
result = model_wiki.most_similar(positive=['king', 'woman'], negative=['man'])
print("\nReal Model Result:")
for word, score in result[:5]:
    print(f"{word}: {score:.4f}")
