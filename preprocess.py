import ftfy
import phonenumbers
import spacy
import unicodedata
import re

TECHNICAL_PHRASES = {}
with open('technicals.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if '|' in line:
            phrase, replacement = line.strip().split('|')
            TECHNICAL_PHRASES[phrase] = replacement

SECTION_HEADERS = set()
with open('sections.txt', 'r', encoding='utf-8') as f:
    SECTION_HEADERS = {line.strip() for line in f if line.strip()}

def load_spacy_model():
    nlp = spacy.load("en_core_web_lg")
    return nlp

def fix_unicode_issues(text):
    return ftfy.fix_text(text)

def remove_phone_numbers(text):
    cleaned = text
    for match in phonenumbers.PhoneNumberMatcher(cleaned, "PK"):
        cleaned = cleaned.replace(match.raw_string, " ")
    for match in phonenumbers.PhoneNumberMatcher(cleaned, None):
        cleaned = cleaned.replace(match.raw_string, " ")
    return cleaned

def remove_contact_info(text, nlp):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.like_email or token.like_url:
            continue
        tokens.append(token.text)
    return " ".join(tokens)

def remove_symbols_and_special_chars(text):
    cleaned = "".join(
        ch if unicodedata.category(ch)[0] not in ['S', 'C'] else " "
        for ch in text
    )
    cleaned = re.sub(r'[|•●○■□▪▫→←↑↓►▼◄▲▼◆◇]', ' ', cleaned)
    return cleaned

def remove_person_names(text, nlp):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.ent_type_ == "PERSON":
            continue
        tokens.append(token.text)
    return " ".join(tokens)

def normalize_whitespace(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preserve_technical_phrases(text):
    text_lower = text.lower()
    sorted_phrases = sorted(TECHNICAL_PHRASES.items(), key=lambda x: len(x[0]), reverse=True)
    for phrase, replacement in sorted_phrases:
        text_lower = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text_lower)
    return text_lower

def tokenize_lemmatize_stopwords(text, nlp):
    doc = nlp(text)
    tokens = []
    
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        if not (token.is_alpha or '_' in token.text):
            continue
        if len(token.text) < 2:
            continue
        
        lemma = token.lemma_.lower()
        
        if lemma in SECTION_HEADERS:
            continue
        
        tokens.append(lemma)
    
    return " ".join(tokens)


def preprocess_text(text, nlp):
    text = fix_unicode_issues(text)
    text = remove_phone_numbers(text)
    text = remove_contact_info(text, nlp)
    text = remove_symbols_and_special_chars(text)
    text = remove_person_names(text, nlp)
    text = normalize_whitespace(text)
    text = preserve_technical_phrases(text)
    text = tokenize_lemmatize_stopwords(text, nlp)
    return text


if __name__ == "__main__":
    import pandas as pd
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py [resumes|jds]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    nlp = load_spacy_model()
    
    if mode == "resumes":
        df = pd.read_csv("Dataset/resumes.csv")
        print(f"Loaded {len(df):,} resumes")
        
        cleaned_texts = []
        for text in df['resume_text']:
            cleaned = preprocess_text(text, nlp)
            cleaned_texts.append(cleaned)
        
        df['resume_text_cleaned'] = cleaned_texts
        output_file = "resumes_cleaned.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to '{output_file}'")
    
    elif mode == "jds":
        df = pd.read_csv("job_descriptions.csv")
        print(f"Loaded {len(df):,} job descriptions")
        
        cleaned_texts = []
        for text in df['job_description']:
            cleaned = preprocess_text(text, nlp)
            cleaned_texts.append(cleaned)
        
        df['job_description_cleaned'] = cleaned_texts
        output_file = "job_descriptions_cleaned.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to '{output_file}'")
    
    else:
        print("Invalid mode. Use 'resumes' or 'jds'")
