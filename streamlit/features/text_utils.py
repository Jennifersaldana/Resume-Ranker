import pdfplumber
import re

STOPWORDS = {
    "the", "and", "a", "an", "of", "to", "in", "for", "on", "with", "by", "at",
    "as", "is", "are", "be", "this", "that", "from", "or", "it", "its", "their",
    "our", "we", "they", "you", "your", "such", "including", "etc", "will",
    "responsible", "responsibilities", "must", "should", "have", "has", "had"
}

def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text.strip()

def tokenize_words(text: str):
    tokens = re.findall(r"[A-Za-z\+\#\.\-]{2,}", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

def extract_skills(text: str):
    tokens = tokenize_words(text)
    skills = set()
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            skills.add(t)
        elif len(t) > 3:
            skills.add(t)
    return skills
