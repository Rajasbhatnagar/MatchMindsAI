from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from agents.resume_parser import parse_resume

class ResumeMatchingModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = SVC(kernel='linear', probability=True)
        self.scaler = StandardScaler(with_mean=False)
        nltk.download('punkt')
        nltk.download('stopwords')
        
    def preprocess_text(self, text):
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)
        
    def prepare_data(self, resume_paths, labels):
        texts = []
        for path in resume_paths:
            resume_text = parse_resume(path)
            processed_text = self.preprocess_text(resume_text)
            texts.append(processed_text)
        
        X = self.vectorizer.fit_transform(texts)
        X = self.scaler.fit_transform(X)
        return X, labels
    
    def tune_hyperparameters(self, X, y):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5)
        grid_search.fit(X, y)
        self.classifier = grid_search.best_estimator_
        
    def train(self, X, y):
        # Perform cross-validation
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f}")
        
        # Train the final model
        self.classifier.fit(X, y)
        
    def evaluate(self, X, y_true):
        y_pred = self.classifier.predict(X)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
    
    def predict_proba(self, resume_text):
        processed_text = self.preprocess_text(resume_text)
        X = self.vectorizer.transform([processed_text])
        X = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X)[0]
        return {'match_probability': probs[1], 'non_match_probability': probs[0]}
    
    def save_model(self, path):
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'scaler': self.scaler
        }, path)
    
    def load_model(self, path):
        model_data = joblib.load(path)
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
