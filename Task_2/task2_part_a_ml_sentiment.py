# Task_2/task2_part_a_ml_sentiment.py

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("TASK 2 - PART A: MACHINE LEARNING FOR TEXT CLASSIFICATION")
print("Dataset: IMDb Reviews")
print("="*60)

# 1. Load Dataset
print("\n[1] Loading IMDb dataset...")
# Using tensorflow_datasets to get raw text
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['test']

# Convert to lists
texts = []
labels = []
for text, label in tfds.as_numpy(train_data):
    texts.append(text.decode('utf-8'))
    labels.append(label)
for text, label in tfds.as_numpy(test_data):
    texts.append(text.decode('utf-8'))
    labels.append(label)

texts = np.array(texts)
labels = np.array(labels)

print(f"Total samples: {len(texts)}")

# 2. Report Class Distribution
print("\n[2] Class Distribution:")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} ({'Positive' if u==1 else 'Negative'}): {c} samples ({c/len(labels)*100:.2f}%)")

# 3. Text Preprocessing
print("\n[3] Preprocessing text data...")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Clean all texts
cleaned_texts = [clean_text(text) for text in texts]
print("Text cleaning completed.")

# 4. Fixed Train/Test Split
print("\n[4] Creating fixed train/test split (random_state=42)...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    cleaned_texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save the split data for Part B to ensure exact same split
os.makedirs('Task_2/data', exist_ok=True)
with open('Task_2/data/cleaned_texts.pkl', 'wb') as f:
    pickle.dump(cleaned_texts, f)
with open('Task_2/data/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
# Save indices to reconstruct exact split in Part B
np.save('Task_2/data/train_indices.npy', np.where(np.isin(cleaned_texts, X_train_text))[0]) # Simplified approach: just save the arrays
# Better approach: save the actual split arrays
with open('Task_2/data/X_train_text.pkl', 'wb') as f:
    pickle.dump(X_train_text, f)
with open('Task_2/data/X_test_text.pkl', 'wb') as f:
    pickle.dump(X_test_text, f)
with open('Task_2/data/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('Task_2/data/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

print(f"Training set: {len(X_train_text)} samples")
print(f"Test set: {len(X_test_text)} samples")

# 5. TF-IDF Vectorization
print("\n[5] Applying TF-IDF vectorization...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

# Save vectorizer
with open('Task_2/data/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# 6. Train Logistic Regression
print("\n[6] Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
lr_model.fit(X_train_tfidf, y_train)

# 7. Train Support Vector Machine (SVM)
print("\n[7] Training Support Vector Machine (SVM)...")
svm_model = SVC(random_state=42, kernel='linear', C=1.0)
svm_model.fit(X_train_tfidf, y_train)

# 8. Evaluate Models
print("\n[8] Evaluating models...")

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'confusion_matrix': cm}

lr_metrics = evaluate_model(lr_model, X_test_tfidf, y_test, "Logistic Regression")
svm_metrics = evaluate_model(svm_model, X_test_tfidf, y_test, "SVM")

# 9. Save Results
print("\n[9] Saving results...")
os.makedirs('Task_2/results', exist_ok=True)

results = {
    'Logistic Regression': {k: float(v) if k != 'confusion_matrix' else v.tolist() for k, v in lr_metrics.items()},
    'SVM': {k: float(v) if k != 'confusion_matrix' else v.tolist() for k, v in svm_metrics.items()}
}

with open('Task_2/results/ml_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

np.save('Task_2/results/lr_confusion_matrix.npy', lr_metrics['confusion_matrix'])
np.save('Task_2/results/svm_confusion_matrix.npy', svm_metrics['confusion_matrix'])

# 10. Plot Confusion Matrices
print("\n[10] Generating plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(svm_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('Task_2/results/ml_confusion_matrices.png', dpi=300)
plt.close()

print("\n" + "="*60)
print("✅ TASK 2 PART A COMPLETED SUCCESSFULLY!")
print("="*60)
