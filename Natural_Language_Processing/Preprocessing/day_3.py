from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

corpus = [
    "The AI engineer is coding Python.",
    "The data scientist is coding SQL.",
    "Python is great for AI.",
    "I love coding."
]

print(f"Corpus: {corpus}\n")

# --- Method 1: Bag of Words (BoW) ---
print("--- 1. Bag of Words (CountVectorizer) ---")
# stops_words='english' removes "is", "the", "for" automatically
bow_vectorizer = CountVectorizer(stop_words='english')
bow_matrix = bow_vectorizer.fit_transform(corpus)

# Convert to readable DataFrame
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
print(bow_df)
print("\n")

# --- Method 2: TF-IDF ---
print("--- 2. TF-IDF (Weighted) ---")
tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
tf_idf_matrix = tf_idf_vectorizer.fit_transform(corpus)

tf_idf_df = pd.DataFrame(tf_idf_matrix.toarray(), columns=tf_idf_vectorizer.get_feature_names_out())
print(tf_idf_df.round(2))
