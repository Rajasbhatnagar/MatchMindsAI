import fitz  # PyMuPDF
from textblob import TextBlob
import re
import logging

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

def parse_resume(file_path, correct_spelling_enabled=True):
    try:
        with fitz.open(file_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        
        if not text.strip():
            logging.warning(f"Warning: No text extracted from {file_path}")
            return ""
            
        text = clean_text(text)
        if correct_spelling_enabled:
            text = correct_spelling(text)
        return text
    except Exception as e:
        logging.error(f"Error parsing resume {file_path}: {str(e)}")
        return ""