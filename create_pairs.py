import pandas as pd
import numpy as np
import os

print("Loading cleaned datasets...")
resumes_df = pd.read_csv("resumes_cleaned.csv")
jobs_df = pd.read_csv("job_descriptions_cleaned.csv")

print(f"Loaded {len(resumes_df):,} resumes, {len(jobs_df):,} job descriptions")

print("Verifying category alignment...")
resume_categories = set(resumes_df['resume_category'].unique())
jd_categories = set(jobs_df['job_category'].unique())

categories = sorted(resume_categories & jd_categories)
print(f"Using {len(categories)} common categories")

print("Generating resume-job pairs...")

pairs = []
np.random.seed(42)

positive_count = 0
negative_count = 0

jd_by_category = {cat: jobs_df[jobs_df['job_category'] == cat] for cat in categories}

for idx, resume_row in resumes_df.iterrows():
    resume_id = resume_row['resume_id']
    resume_category = resume_row['resume_category']
    resume_text = resume_row['resume_text_cleaned']
    
    if resume_category not in categories:
        continue
    
    same_category_jds = jd_by_category[resume_category]
    other_categories = [cat for cat in categories if cat != resume_category]
    for _, jd_row in same_category_jds.iterrows():
        pairs.append({
            'resume_id': resume_id,
            'job_id': jd_row['job_id'],
            'resume_category': resume_category,
            'job_category': jd_row['job_category'],
            'resume_text_cleaned': resume_text,
            'job_description_cleaned': jd_row['job_description_cleaned'],
            'label': 1
        })
        positive_count += 1
    
    num_positive = len(same_category_jds)
    sampled_negative_jds = []
    for other_cat in other_categories:
        other_jds = jd_by_category[other_cat]
        sampled_negative_jds.append(other_jds)
    
    all_other_jds = pd.concat(sampled_negative_jds, ignore_index=True)
    
    if len(all_other_jds) >= num_positive:
        sampled_jds = all_other_jds.sample(n=num_positive, random_state=idx)
    else:
        sampled_jds = all_other_jds.sample(n=num_positive, replace=True, random_state=idx)
    
    for _, jd_row in sampled_jds.iterrows():
        pairs.append({
            'resume_id': resume_id,
            'job_id': jd_row['job_id'],
            'resume_category': resume_category,
            'job_category': jd_row['job_category'],
            'resume_text_cleaned': resume_text,
            'job_description_cleaned': jd_row['job_description_cleaned'],
            'label': 0
        })
        negative_count += 1
    
    if (idx + 1) % 500 == 0 or (idx + 1) == len(resumes_df):
        print(f"Progress: {(idx+1)/len(resumes_df)*100:.1f}%", end="\r")

print(f"\nGenerated {len(pairs):,} pairs ({positive_count:,} positive, {negative_count:,} negative)")

pairs_df = pd.DataFrame(pairs)
pairs_df = pairs_df.sample(frac=1, random_state=42).reset_index(drop=True)

output_file = "dataset_pairs.csv"
pairs_df.to_csv(output_file, index=False)
file_size = os.path.getsize(output_file) / (1024*1024)
print(f"Saved to '{output_file}' ({file_size:.2f} MB)")

label_counts = pairs_df['label'].value_counts().sort_index()
balance_ratio = min(label_counts[0], label_counts[1]) / max(label_counts[0], label_counts[1])

print(f"\nDataset Statistics:")
print(f"  Total pairs: {len(pairs_df):,}")
print(f"  Unique resumes: {pairs_df['resume_id'].nunique():,}")
print(f"  Unique jobs: {pairs_df['job_id'].nunique():,}")
print(f"  Label 0: {label_counts[0]:,}, Label 1: {label_counts[1]:,}")
print(f"  Balance ratio: {balance_ratio:.3f}")
print(f"\nPair creation complete!")
