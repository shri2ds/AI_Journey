import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Spacy for cleaning
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
        Standard cleaning pipeline: Lowercase -> Tokenize -> Lemmatize -> Remove Stop Words
    """
    doc = nlp(text.lower())
    clean_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            clean_tokens.append(token.lemma_)
    return " ".join(clean_tokens)

def calculate_match(resume, job_description):
    # Clean both texts
    clean_resume = clean_text(resume)
    clean_jd = clean_text(job_description)

    # Vectorizer
    vectorizer = CountVectorizer()

    # Fit on both strings to build a common vocabulary
    vectors = vectorizer.fit_transform([clean_resume, clean_jd])

    # Calculate Cosine Similarity
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Extract Missing Keywords
    feature_names = vectorizer.get_feature_names_out()

    # Convert sparse vectors to standard arrays
    resume_vector = vectors[0].toarray()[0]
    jd_vector = vectors[1].toarray()[0]

    missing_keywords = []
    for i, word in enumerate(feature_names):
        if jd_vector[i] > 0 and resume_vector[i] == 0:
            missing_keywords.append(word)

    return round(similarity*100, 2),  missing_keywords

# --- Test Case ---
my_resume = """
I am an enthusiastic Data Scientist with experience in Python, Pandas, and Scikit-Learn.
I have built machine learning models for classification.
"""

target_job = """
We are looking for a Senior Data Scientist.
Must have strong skills in Python, Scikit-Learn, and SQL.
Experience with Deep Learning (PyTorch) is a plus.
"""

score, missing = calculate_match(my_resume, target_job)

print(f"Match Score: {score}%")
print(f"Missing Keywords: {missing}")
