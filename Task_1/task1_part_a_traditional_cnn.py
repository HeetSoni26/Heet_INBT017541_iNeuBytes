# Task_1/task1_part_a_traditional_cnn.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("TASK 1 - PART A: TRADITIONAL CNN MODEL")
print("CIFAR-10 Image Classification")
print("="*60)

# 1. Load and Preprocess CIFAR-10 Dataset
print("\n[1] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")

# 2. Normalize pixel values (0-255 -> 0-1)
print("\n[2] Normalizing pixel values...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Create validation set (use same split for Part B)
print("\n[3] Creating train/validation/test split...")
# Split 10% for validation
val_size = int(0.1 * len(x_train))
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train_final = x_train[:-val_size]
y_train_final = y_train[:-val_size]

print(f"Training set: {x_train_final.shape[0]} samples")
print(f"Validation set: {x_val.shape[0]} samples")
print(f"Test set: {x_test.shape[0]} samples")

# 4. Build Traditional AlexNet-style CNN (adapted for 32x32 images)
print("\n[4] Building Traditional CNN Architecture...")

def build_traditional_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    AlexNet-style CNN adapted for CIFAR-10 (32x32 images)
    Original AlexNet was built for 227x227 inputs
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512),
        layers.Activation('relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes),
        layers.Activation('softmax')
    ])
    
    return model

# Create the model
model_traditional = build_traditional_cnn()

# Display model summary
print("\nModel Architecture Summary:")
print("="*60)
model_traditional.summary()

# Count parameters
total_params = model_traditional.count_params()
print(f"\nTotal Parameters: {total_params:,}")

# 5. Compile the Model
print("\n[5] Compiling the model...")
model_traditional.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Define Callbacks
print("\n[6] Setting up callbacks...")

# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce Learning Rate on Plateau
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# 7. Train the Model
print("\n[7] Training the model...")
print("="*60)

# Training parameters
EPOCHS = 50
BATCH_SIZE = 64

history = model_traditional.fit(
    x_train_final, y_train_final,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 8. Evaluate on Test Set
print("\n[8] Evaluating on test set...")
test_loss, test_accuracy = model_traditional.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# 9. Make Predictions
print("\n[9] Generating predictions...")
y_pred_proba = model_traditional.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# 10. Calculate Metrics
print("\n[10] Calculating performance metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n{'='*60}")
print("PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"{'='*60}")

# 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 12. Confusion Matrix
print("\n[11] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

# 13. Save Results
print("\n[12] Saving results...")

# Create results directory
os.makedirs('Task_1/results', exist_ok=True)

# Save metrics to JSON
metrics = {
    'model_type': 'Traditional CNN',
    'test_accuracy': float(test_accuracy),
    'test_loss': float(test_loss),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'total_parameters': int(total_params),
    'epochs_trained': len(history.history['loss']),
    'class_names': class_names
}

with open('Task_1/results/traditional_cnn_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save confusion matrix
np.save('Task_1/results/traditional_cnn_confusion_matrix.npy', cm)

# Save model
model_traditional.save('Task_1/results/traditional_cnn_model.h5')
print("Model saved to Task_1/results/traditional_cnn_model.h5")

# 14. Plot Training History
print("\n[13] Plotting training history...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, 
            ax=axes[1, 0], cbar_kws={'shrink': 0.8})
axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
plt.setp(axes[1, 0].get_yticklabels(), rotation=0)

# Training vs Validation Accuracy Comparison
epochs_range = range(1, len(history.history['accuracy']) + 1)
axes[1, 1].plot(epochs_range, history.history['accuracy'], 'o-', 
                label='Training', linewidth=2, markersize=5)
axes[1, 1].plot(epochs_range, history.history['val_accuracy'], 's-', 
                label='Validation', linewidth=2, markersize=5)
axes[1, 1].axhline(y=0.70, color='r', linestyle='--', label='Target (70%)', linewidth=2)
axes[1, 1].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Task_1/results/traditional_cnn_training_history.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots saved to Task_1/results/traditional_cnn_training_history.png")

# 15. Display Sample Predictions
print("\n[14] Displaying sample predictions...")

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i in range(12):
    idx = np.random.randint(0, len(x_test))
    axes[i].imshow(x_test[idx])
    pred_class = class_names[y_pred[idx]]
    true_class = class_names[y_test[idx][0]]
    color = 'green' if y_pred[idx] == y_test[idx][0] else 'red'
    axes[i].set_title(f'Pred: {pred_class}\nTrue: {true_class}', 
                      color=color, fontsize=8, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('Task_1/results/traditional_cnn_sample_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Sample predictions saved to Task_1/results/traditional_cnn_sample_predictions.png")

print("\n" + "="*60)
print("✅ TASK 1 PART A COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Final Results:")
print(f"   • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Target (≥70%): {'✅ ACHIEVED' if test_accuracy >= 0.70 else '❌ NOT ACHIEVED'}")
print(f"   • Total Parameters: {total_params:,}")
print(f"   • Epochs Trained: {len(history.history['loss'])}")
print("\n📁 All results saved in Task_1/results/")
print("="*60)
