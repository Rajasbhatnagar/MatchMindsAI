import os
from sklearn.model_selection import train_test_split
from agents.resume_parser import parse_resume
from agents.model_trainer import ResumeMatchingModel
import numpy as np

def evaluate_resumes(jd_keywords, resume_paths, test_size=0.2):
    # Split data into train and test sets
    train_paths, test_paths = train_test_split(resume_paths, test_size=test_size, random_state=42)
    
    # Process all resumes to create feature vectors
    all_texts = []
    for path in train_paths:
        resume_text = parse_resume(path) 
        all_texts.append(resume_text)
    
    # Generate initial labels based on keyword matching
    keyword_scores = []
    for text in all_texts:
        matches = sum(1 for kw in jd_keywords if kw.lower() in text.lower())
        score = matches / len(jd_keywords)
        keyword_scores.append(score)
    
    # Convert scores to binary labels using median as threshold
    threshold = np.median(keyword_scores)
    labels = [1 if score >= threshold else 0 for score in keyword_scores]
    
    # Train model
    model = ResumeMatchingModel()
    X, y = model.prepare_data(train_paths, labels)
    model.train(X, y)
    
    results = {"train": {}, "test": {}}
    
    # Evaluate
    for category, paths in [("train", train_paths), ("test", test_paths)]:
        for path in paths:
            name = os.path.basename(path).replace(".pdf", "")
            resume_text = parse_resume(path)
            score = model.predict_proba(resume_text)
            results[category][name] = round(float(score), 3)
    
    return results