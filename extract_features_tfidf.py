import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

pairs_df = pd.read_csv("dataset_pairs.csv")
print(f"Loaded {len(pairs_df):,} pairs")

pairs_df['combined_text'] = (
    pairs_df['resume_text_cleaned'] + ' [SEP] ' + pairs_df['job_description_cleaned']
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    sublinear_tf=True,
    strip_accents='unicode'
)

X_tfidf = vectorizer.fit_transform(pairs_df['combined_text'])
y_labels = pairs_df['label'].values

print(f"Feature matrix: {X_tfidf.shape}")
print(f"Labels: Positive={(y_labels == 1).sum():,}, Negative={(y_labels == 0).sum():,}")

with open("models/tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(vectorizer, f)

print("Saved models/tfidf_vectorizer.pkl")
