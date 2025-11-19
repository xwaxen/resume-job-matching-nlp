import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

pairs_df = pd.read_csv("dataset_pairs.csv")
print(f"Loaded {len(pairs_df):,} pairs")

resume_texts = pairs_df['resume_text_cleaned'].tolist()
jd_texts = pairs_df['job_description_cleaned'].tolist()

print("Encoding resumes...")
resume_embeddings = model.encode(
    resume_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Encoding job descriptions...")
jd_embeddings = model.encode(
    jd_texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"Resume embeddings: {resume_embeddings.shape}")
print(f"JD embeddings: {jd_embeddings.shape}")
