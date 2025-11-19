import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path
import PyPDF2
from docx import Document

RESUME_PATHS = ["test/Shahzaib-Resume.pdf","test/Kumel_Resume.pdf"]
JOB_DESC_PATH = "test/Description.txt"


def preprocess(text):
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '', text)
    
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    text = re.sub(r'[•●★☆■□▪▫◆◇○◉→←↑↓]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    text = text.lower()
    
    return text


def preprocess_resume(resume_text):
    return preprocess(resume_text)


def preprocess_job_description(jd_text):
    return preprocess(jd_text)  


def load_text_file(file_path):
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return text
    
    elif file_ext == '.pdf':
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text.strip()
    
    elif file_ext == '.docx':
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .txt, .pdf, or .docx")


while True:
    print("Select Model:")
    print("1) Logistic Regression")
    print("2) XGBoost")
    print("3) Both")
    model_choice = input("Enter choice (1/2/3): ").strip()
    
    if model_choice in ['1', '2', '3']:
        break
    else:
        print("Invalid input. Try again.\n")

print("\nLoading models...")

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

logreg_model = None
tfidf_vectorizer = None
xgb_model = None
sbert_model = None

if model_choice in ['1', '3']:
    with open('models/logreg_model.pkl', 'rb') as f:
        logreg_model = pickle.load(f)
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

if model_choice in ['2', '3']:
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
        xgb_model.set_params(device='cpu')
    device = 'cpu'
    sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

print("Loading job description...")
job_desc_text = load_text_file(JOB_DESC_PATH)
job_desc_clean = preprocess_job_description(job_desc_text)
jd_emb = sbert_model.encode([job_desc_clean], convert_to_numpy=True, show_progress_bar=False)[0]

if sbert_model is not None:
    jd_emb = sbert_model.encode([job_desc_clean], convert_to_numpy=True, show_progress_bar=False)[0]

print(f"Processing {len(RESUME_PATHS)} resume(s)...\n")

results_logreg = []
results_xgb = []

for resume_path in RESUME_PATHS:
    resume_text = load_text_file(resume_path)
    resume_clean = preprocess_resume(resume_text)
    
    if logreg_model is not None:
        combined_text = resume_clean + " " + job_desc_clean
        tfidf_features = tfidf_vectorizer.transform([combined_text])
        logreg_pred = logreg_model.predict(tfidf_features)[0]
        logreg_proba = logreg_model.predict_proba(tfidf_features)[0]
        logreg_confidence = logreg_proba[logreg_pred] * 100
        
        results_logreg.append({
            'resume_path': resume_path,
            'prediction': logreg_pred,
            'confidence': logreg_confidence
        })
    
    if xgb_model is not None:
        resume_emb = sbert_model.encode([resume_clean], convert_to_numpy=True, show_progress_bar=False)[0]
        diff_emb = resume_emb - jd_emb
        cosine_sim = np.dot(resume_emb, jd_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb))
        sbert_features = np.concatenate([resume_emb, jd_emb, diff_emb, [cosine_sim]])
        sbert_features = sbert_features.reshape(1, -1).astype(np.float32)
        
        xgb_pred = xgb_model.predict(sbert_features)[0]
        xgb_proba = xgb_model.predict_proba(sbert_features)[0]
        xgb_confidence = xgb_proba[1] * 100
        
        results_xgb.append({
            'resume_path': resume_path,
            'prediction': xgb_pred,
            'confidence': xgb_confidence
        })

if model_choice == '1':
    results_logreg.sort(key=lambda x: x['confidence'], reverse=True)
    good_fits = [r for r in results_logreg if r['prediction'] == 1]
    not_fits = [r for r in results_logreg if r['prediction'] == 0]
    
    print("=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)
    print("GOOD FIT:\n")
    if good_fits:
        for r in good_fits:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    
    print("\n" + "=" * 60)
    print("NOT FIT:\n")
    if not_fits:
        for r in not_fits:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    print("=" * 60)

elif model_choice == '2':
    results_xgb.sort(key=lambda x: x['confidence'], reverse=True)
    good_fits = [r for r in results_xgb if r['prediction'] == 1]
    not_fits = [r for r in results_xgb if r['prediction'] == 0]
    
    print("=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)
    print("GOOD FIT:\n")
    if good_fits:
        for r in good_fits:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    
    print("\n" + "=" * 60)
    print("NOT FIT:\n")
    if not_fits:
        for r in not_fits:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    print("=" * 60)

else:
    results_logreg.sort(key=lambda x: x['confidence'], reverse=True)
    results_xgb.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)
    good_fits_lr = [r for r in results_logreg if r['prediction'] == 1]
    not_fits_lr = [r for r in results_logreg if r['prediction'] == 0]
    
    print("GOOD FIT:\n")
    if good_fits_lr:
        for r in good_fits_lr:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    
    print("\n" + "=" * 60)
    print("NOT FIT:\n")
    if not_fits_lr:
        for r in not_fits_lr:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    
    print("\n" + "=" * 60)
    print("XGBOOST RESULTS")
    print("=" * 60)
    good_fits_xgb = [r for r in results_xgb if r['prediction'] == 1]
    not_fits_xgb = [r for r in results_xgb if r['prediction'] == 0]
    
    print("GOOD FIT:\n")
    if good_fits_xgb:
        for r in good_fits_xgb:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    
    print("\n" + "=" * 60)
    print("NOT FIT:\n")
    if not_fits_xgb:
        for r in not_fits_xgb:
            print(f"{Path(r['resume_path']).name:<40} {r['confidence']:.1f}%")
    else:
        print("None")
    print("=" * 60)