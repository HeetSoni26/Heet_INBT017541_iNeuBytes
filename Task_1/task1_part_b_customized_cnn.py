# Task_1/task1_part_b_customized_cnn.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import os

# Set random seeds (MUST match Part A)
np.random.seed(42)
tf.random.set_seed(42)

print("="*60)
print("TASK 1 - PART B: CUSTOMIZED CNN MODEL")
print("Improving upon Traditional CNN")
print("="*60)

# 1. Load CIFAR-10 Dataset (same as Part A)
print("\n[1] Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Normalize pixel values
print("\n[2] Normalizing pixel values...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Create validation set (SAME split as Part A)
print("\n[3] Creating train/validation/test split (same as Part A)...")
val_size = int(0.1 * len(x_train))
x_val = x_train[-val_size:]
y_val = y_train[-val_size:]
x_train_final = x_train[:-val_size]
y_train_final = y_train[:-val_size]

# 4. Data Augmentation (NEW in Part B)
print("\n[4] Setting up data augmentation...")
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# 5. Build Customized CNN
print("\n[5] Building Customized CNN Architecture...")

def build_customized_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Customized CNN with improvements over traditional model:
    - More convolutional layers
    - Different filter sizes
    - Batch normalization
    - Dropout
    - Data augmentation
    - Optimized learning rate
    """
    model = keras.Sequential([
        # Data Augmentation
        data_augmentation,
        
        # Block 1: 64 filters
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: 128 filters
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        
        # Block 3: 256 filters
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),
        
        # Block 4: 512 filters (NEW)
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Output Layer
        layers.Dense(num_classes),
        layers.Activation('softmax')
    ])
    
    return model

# Create the model
model_customized = build_customized_cnn()

print("\nModel Architecture Summary:")
print("="*60)
model_customized.summary()

# Count parameters
total_params_customized = model_customized.count_params()
print(f"\nTotal Parameters: {total_params_customized:,}")

# 6. Compile the Model
print("\n[6] Compiling the model...")

# Learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)

model_customized.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Define Callbacks
print("\n[7] Setting up callbacks...")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

# 8. Train the Model
print("\n[8] Training the model...")
print("="*60)

# Use SAME number of epochs as Part A for fair comparison
EPOCHS = 50
BATCH_SIZE = 64

history_customized = model_customized.fit(
    x_train_final, y_train_final,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 9. Evaluate on Test Set
print("\n[9] Evaluating on test set...")
test_loss_customized, test_accuracy_customized = model_customized.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy_customized:.4f} ({test_accuracy_customized*100:.2f}%)")
print(f"Test Loss: {test_loss_customized:.4f}")

# 10. Make Predictions
print("\n[10] Generating predictions...")
y_pred_proba_customized = model_customized.predict(x_test)
y_pred_customized = np.argmax(y_pred_proba_customized, axis=1)

# 11. Calculate Metrics
print("\n[11] Calculating performance metrics...")
accuracy_customized = accuracy_score(y_test, y_pred_customized)
precision_customized = precision_score(y_test, y_pred_customized, average='weighted')
recall_customized = recall_score(y_test, y_pred_customized, average='weighted')
f1_customized = f1_score(y_test, y_pred_customized, average='weighted')

print(f"\n{'='*60}")
print("CUSTOMIZED CNN PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Accuracy:  {accuracy_customized:.4f} ({accuracy_customized*100:.2f}%)")
print(f"Precision: {precision_customized:.4f}")
print(f"Recall:    {recall_customized:.4f}")
print(f"F1-Score:  {f1_customized:.4f}")
print(f"{'='*60}")

# 12. Load Traditional CNN Results for Comparison
print("\n[12] Loading Traditional CNN results for comparison...")
with open('Task_1/results/traditional_cnn_metrics.json', 'r') as f:
    traditional_metrics = json.load(f)

traditional_accuracy = traditional_metrics['test_accuracy']
improvement = test_accuracy_customized - traditional_accuracy
improvement_percent = (improvement / traditional_accuracy) * 100

print(f"\n{'='*60}")
print("COMPARISON: TRADITIONAL vs CUSTOMIZED CNN")
print(f"{'='*60}")
print(f"Traditional CNN Accuracy: {traditional_accuracy:.4f} ({traditional_accuracy*100:.2f}%)")
print(f"Customized CNN Accuracy:  {test_accuracy_customized:.4f} ({test_accuracy_customized*100:.2f}%)")
print(f"Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")
print(f"Target (≥3% improvement): {'✅ ACHIEVED' if improvement >= 0.03 else '❌ NOT ACHIEVED'}")
print(f"{'='*60}")

# 13. Classification Report
print("\nClassification Report (Customized CNN):")
print(classification_report(y_test, y_pred_customized, target_names=class_names))

# 14. Confusion Matrix
print("\n[13] Generating confusion matrix...")
cm_customized = confusion_matrix(y_test, y_pred_customized)

# 15. Save Results
print("\n[14] Saving results...")

os.makedirs('Task_1/results', exist_ok=True)

# Save metrics
metrics_customized = {
    'model_type': 'Customized CNN',
    'test_accuracy': float(test_accuracy_customized),
    'test_loss': float(test_loss_customized),
    'accuracy': float(accuracy_customized),
    'precision': float(precision_customized),
    'recall': float(recall_customized),
    'f1_score': float(f1_customized),
    'total_parameters': int(total_params_customized),
    'epochs_trained': len(history_customized.history['loss']),
    'improvement_over_traditional': float(improvement),
    'improvement_percentage': float(improvement_percent),
    'class_names': class_names
}

with open('Task_1/results/customized_cnn_metrics.json', 'w') as f:
    json.dump(metrics_customized, f, indent=2)

# Save confusion matrix
np.save('Task_1/results/customized_cnn_confusion_matrix.npy', cm_customized)

# Save model
model_customized.save('Task_1/results/customized_cnn_model.h5')
print("Model saved to Task_1/results/customized_cnn_model.h5")

# 16. Plot Comparison
print("\n[15] Plotting comparison...")

# Load traditional model history (if available)
try:
    traditional_model = keras.models.load_model('Task_1/results/traditional_cnn_model.h5')
except:
    traditional_model = None

fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# 1. Traditional CNN Accuracy
if traditional_model:
    axes[0, 0].plot([test_accuracy_customized - improvement], [traditional_accuracy], 
                    'o', markersize=15, color='blue', label='Traditional')
    axes[0, 0].set_title('Traditional CNN Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].axhline(y=0.70, color='r', linestyle='--', label='Target (70%)')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

# 2. Customized CNN Accuracy
axes[0, 1].plot([test_accuracy_customized], 'o', markersize=15, color='green', label='Customized')
axes[0, 1].set_title('Customized CNN Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].axhline(y=0.70, color='r', linestyle='--', label='Target (70%)')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix - Traditional
if traditional_model:
    cm_trad = np.load('Task_1/results/traditional_cnn_confusion_matrix.npy')
    sns.heatmap(cm_trad, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1, 0], cbar_kws={'shrink': 0.8})
    axes[1, 0].set_title('Confusion Matrix - Traditional CNN', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# 4. Confusion Matrix - Customized
sns.heatmap(cm_customized, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1, 1], cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Confusion Matrix - Customized CNN', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

# 5. Parameter Comparison
models = ['Traditional', 'Customized']
params = [traditional_metrics['total_parameters'], total_params_customized]
colors = ['blue', 'green']
axes[2, 0].bar(models, params, color=colors, alpha=0.7)
axes[2, 0].set_title('Parameter Count Comparison', fontsize=12, fontweight='bold')
axes[2, 0].set_ylabel('Number of Parameters')
for i, v in enumerate(params):
    axes[2, 0].text(i, v + 10000, f'{v:,}', ha='center', fontweight='bold')
axes[2, 0].grid(True, alpha=0.3, axis='y')

# 6. Accuracy Comparison
accs = [traditional_accuracy, test_accuracy_customized]
axes[2, 1].bar(models, accs, color=colors, alpha=0.7)
axes[2, 1].axhline(y=0.70, color='r', linestyle='--', label='Target (70%)', linewidth=2)
axes[2, 1].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
axes[2, 1].set_ylabel('Accuracy')
axes[2, 1].legend()
for i, v in enumerate(accs):
    axes[2, 1].text(i, v + 0.01, f'{v*100:.2f}%', ha='center', fontweight='bold')
axes[2, 1].set_ylim([0, 1])
axes[2, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Task_1/results/cnn_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Comparison plot saved to Task_1/results/cnn_comparison.png")

# 17. Save Architecture Diagram Description
print("\n[16] Saving architecture descriptions...")

architecture_description = {
    'traditional_cnn': {
        'blocks': 3,
        'conv_layers': [64, 128, 256],
        'dense_layers': [512],
        'dropout_rates': [0.25, 0.25, 0.25, 0.5],
        'data_augmentation': False,
        'batch_normalization': True,
        'learning_rate': 0.001
    },
    'customized_cnn': {
        'blocks': 4,
        'conv_layers': [64, 64, 128, 128, 256, 256, 512, 512],
        'dense_layers': [512, 256],
        'dropout_rates': [0.25, 0.3, 0.3, 0.4, 0.5, 0.4],
        'data_augmentation': True,
        'batch_normalization': True,
        'learning_rate': 'Exponential Decay (0.001)',
        'improvements': [
            'Added more convolutional layers (8 vs 3)',
            'Increased network depth',
            'Data augmentation (flip, rotation, zoom, contrast)',
            'Higher dropout rates for regularization',
            'Additional dense layer',
            'Learning rate scheduling'
        ]
    }
}

with open('Task_1/results/architecture_description.json', 'w') as f:
    json.dump(architecture_description, f, indent=2)

print("\n" + "="*60)
print("✅ TASK 1 PART B COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Final Results:")
print(f"   • Traditional CNN Accuracy: {traditional_accuracy*100:.2f}%")
print(f"   • Customized CNN Accuracy:  {test_accuracy_customized*100:.2f}%")
print(f"   • Improvement: {improvement*100:.2f}%")
print(f"   • Target (≥3% improvement): {'✅ ACHIEVED' if improvement >= 0.03 else '❌ NOT ACHIEVED'}")
print(f"   • Total Parameters: {total_params_customized:,}")
print(f"   • Epochs Trained: {len(history_customized.history['loss'])}")
print("\n📁 All results saved in Task_1/results/")
print("="*60)
