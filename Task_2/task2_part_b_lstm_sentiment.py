# Task_2/task2_part_b_lstm_sentiment.py

import numpy as np
import pickle
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set random seeds (MUST match Part A)
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("TASK 2 - PART B: DEEP LEARNING (LSTM) FOR SENTIMENT ANALYSIS")
print("="*60)

# 1. Load Data from Part A (Ensures exact same split)
print("\n[1] Loading preprocessed data from Part A...")
with open('Task_2/data/X_train_text.pkl', 'rb') as f:
    X_train_text = pickle.load(f)
with open('Task_2/data/X_test_text.pkl', 'rb') as f:
    X_test_text = pickle.load(f)
with open('Task_2/data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('Task_2/data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

print(f"Training samples: {len(X_train_text)}")
print(f"Test samples: {len(X_test_text)}")

# 2. Tokenization and Padding
print("\n[2] Tokenizing text and applying padding...")
MAX_WORDS = 10000
MAX_LENGTH = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

print(f"Padded sequence shape: {X_train_pad.shape}")

# Save tokenizer
with open('Task_2/data/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# 3. Build LSTM Model
print("\n[3] Building LSTM Architecture...")

def build_lstm_model():
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_lstm = build_lstm_model()
print("\nModel Summary:")
model_lstm.summary()

total_params = model_lstm.count_params()
print(f"\nTotal Parameters: {total_params:,}")

# 4. Callbacks
print("\n[4] Setting up callbacks...")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 5. Train Model
print("\n[5] Training LSTM model...")
EPOCHS = 10
BATCH_SIZE = 64

history = model_lstm.fit(
    X_train_pad, y_train,
    validation_split=0.1, # Using 10% of train data for validation during training
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 6. Evaluate on Test Set
print("\n[6] Evaluating on test set...")
test_loss, test_accuracy = model_lstm.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 7. Predictions and Metrics
print("\n[7] Calculating metrics...")
y_pred_proba = model_lstm.predict(X_test_pad)
y_pred_lstm = (y_pred_proba > 0.5).astype(int).flatten()

acc_lstm = accuracy_score(y_test, y_pred_lstm)
prec_lstm = precision_score(y_test, y_pred_lstm, average='weighted')
rec_lstm = recall_score(y_test, y_pred_lstm, average='weighted')
f1_lstm = f1_score(y_test, y_pred_lstm, average='weighted')
cm_lstm = confusion_matrix(y_test, y_pred_lstm)

print(f"\n--- LSTM Results ---")
print(f"Accuracy:  {acc_lstm:.4f}")
print(f"Precision: {prec_lstm:.4f}")
print(f"Recall:    {rec_lstm:.4f}")
print(f"F1-Score:  {f1_lstm:.4f}")

# 8. Load Part A Results for Comparison
print("\n[8] Loading Part A results for comparison...")
with open('Task_2/results/ml_metrics.json', 'r') as f:
    ml_metrics = json.load(f)

# 9. Save Results
print("\n[9] Saving results...")
os.makedirs('Task_2/results', exist_ok=True)

lstm_metrics = {
    'accuracy': float(acc_lstm),
    'precision': float(prec_lstm),
    'recall': float(rec_lstm),
    'f1_score': float(f1_lstm),
    'confusion_matrix': cm_lstm.tolist()
}

comparison_table = {
    'Logistic Regression': ml_metrics['Logistic Regression'],
    'SVM': ml_metrics['SVM'],
    'LSTM': lstm_metrics
}

with open('Task_2/results/lstm_metrics.json', 'w') as f:
    json.dump(lstm_metrics, f, indent=2)
    
with open('Task_2/results/comparison_table.json', 'w') as f:
    json.dump(comparison_table, f, indent=2)

np.save('Task_2/results/lstm_confusion_matrix.npy', cm_lstm)

# 10. Plot Training Curves and Comparison
print("\n[10] Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training Accuracy/Loss
axes[0, 0].plot(history.history['accuracy'], label='Train Acc')
axes[0, 0].plot(history.history['val_accuracy'], label='Val Acc')
axes[0, 0].set_title('LSTM Training/Validation Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history.history['loss'], label='Train Loss')
axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
axes[0, 1].set_title('LSTM Training/Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Oranges', ax=axes[1, 0])
axes[1, 0].set_title('LSTM Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# Final Comparison Bar Chart
models = ['Logistic Regression', 'SVM', 'LSTM']
accuracies = [
    ml_metrics['Logistic Regression']['accuracy'],
    ml_metrics['SVM']['accuracy'],
    lstm_metrics['accuracy']
]
f1_scores = [
    ml_metrics['Logistic Regression']['f1_score'],
    ml_metrics['SVM']['f1_score'],
    lstm_metrics['f1_score']
]

x = np.arange(len(models))
width = 0.35

axes[1, 1].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
axes[1, 1].bar(x + width/2, f1_scores, width, label='F1-Score', color='lightcoral')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models, rotation=15, ha='right')
axes[1, 1].set_title('Model Performance Comparison')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Task_2/results/lstm_analysis.png', dpi=300)
plt.close()

print("\n" + "="*60)
print("✅ TASK 2 PART B COMPLETED SUCCESSFULLY!")
print("="*60)
print("\n📊 Final Comparison:")
for model, metrics in comparison_table.items():
    print(f"   {model}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
print("\n📁 All results saved in Task_2/results/")
print("="*60)
