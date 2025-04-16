import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess_jd(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def extract_keywords(text):
    tokens = word_tokenize(text)
    keywords = [word for word in tokens if word not in stop_words and word.isalpha()]
    return list(set(keywords))

def summarize_jd(jd_text):
    cleaned = preprocess_jd(jd_text)
    doc = nlp(cleaned)

    skills = []
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "WORK_OF_ART", "PRODUCT"]:
            skills.append(ent.text.lower())

    keywords = extract_keywords(cleaned)
    return {
        "skills": list(set(skills)),
        "keywords": keywords
    }