from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_match_score(jd_keywords, resume_text):
    jd_text = " ".join(jd_keywords).lower()
    resume_text = resume_text.lower()
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score, 3)