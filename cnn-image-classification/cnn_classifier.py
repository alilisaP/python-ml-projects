import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

#configuration
BATCH_SIZE = 64
EPOCHS = 50
SEED = 42
np.random.seed(SEED)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 1. data - load, split 60/20/20, normalize, augment
print("=" * 70)
print("1. Data loading")

from tensorflow.keras.datasets import cifar10
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = cifar10.load_data()

#combine all data so we can do our own 60/20/20 split
x_all = np.concatenate([x_train_raw, x_test_raw], axis=0)
y_all = np.concatenate([y_train_raw, y_test_raw], axis=0).flatten()

print(f"  Total: {x_all.shape[0]} images, shape {x_all.shape[1:]}, {len(CLASS_NAMES)} classes")

#shuffle and split 60/20/20
idx = np.random.permutation(len(x_all))
x_all, y_all = x_all[idx], y_all[idx]
n = len(x_all)
n_train = int(0.6 * n)
n_val = int(0.2 * n)

x_train, y_train = x_all[:n_train], y_all[:n_train]
x_val,   y_val   = x_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
x_test,  y_test  = x_all[n_train+n_val:], y_all[n_train+n_val:]

print(f"  Split: train={len(x_train)}, val={len(x_val)}, test={len(x_test)}")

#normalize to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

#data augmentation (rotation, flip, shift, zoom)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
)
datagen.fit(x_train)
print("Augmentation: rotation 15°, shift 10%, horizontal flip, zoom 10%")

#show sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample CIFAR-10 Images', fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i])
    ax.set_title(CLASS_NAMES[y_train[i]], fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig('01_sample_images.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_sample_images.png")


#2.CNN models - 3 different depth architectures
print("\n" + "=" * 70)
print("2. CNN model architectures")

def build_model_A():
    #model A: 1 conv block (shallow)
    model = models.Sequential(name='ModelA_1block')
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_model_B():
    #model B: 2 conv blocks (medium)
    model = models.Sequential(name='ModelB_2blocks')
    #block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    #block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    #classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def build_model_C():
    #model C: 3 conv blocks (deep) with progressive dropout
    model = models.Sequential(name='ModelC_3blocks')
    #block 1: 32 filters
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    #block 2: 64 filters
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    #block 3: 128 filters
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    #classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


MODELS = {
    'Model A (1 block)': build_model_A,
    'Model B (2 blocks)': build_model_B,
    'Model C (3 blocks)': build_model_C,
}

for name, builder in MODELS.items():
    m = builder()
    print(f"\n  {name}: {m.count_params():,} parameters")
    m.summary(print_fn=lambda x: None)


#3. training
print("\n" + "=" * 70)
print("3. Training")

histories = {}
trained = {}

for name, builder in MODELS.items():
    print(f"\n  Training {name}...")
    model = builder()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    ]
    h = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=cb,
        verbose=1,
    )
    histories[name] = h
    trained[name] = model
    print(f"  Best val_acc: {max(h.history['val_accuracy']):.4f}")



#training curves (loss & accuracy)
COLORS = ['#1976D2', '#E64A19', '#388E3C']

fig, axes = plt.subplots(len(MODELS), 2, figsize=(14, 4.5 * len(MODELS)))
fig.suptitle('Training & Validation Curves', fontsize=16, fontweight='bold', y=1.01)
for i, (name, h) in enumerate(histories.items()):
    d = h.history
    ep = range(1, len(d['loss']) + 1)
    c = COLORS[i]
    #loss
    axes[i, 0].plot(ep, d['loss'], '-', color=c, lw=2, label='Train Loss')
    axes[i, 0].plot(ep, d['val_loss'], '--', color=c, lw=2, alpha=0.7, label='Val Loss')
    axes[i, 0].set_title(f'{name} — Loss', fontweight='bold')
    axes[i, 0].set_xlabel('Epoch')
    axes[i, 0].set_ylabel('Loss')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    #accuracy
    axes[i, 1].plot(ep, d['accuracy'], '-', color=c, lw=2, label='Train Acc')
    axes[i, 1].plot(ep, d['val_accuracy'], '--', color=c, lw=2, alpha=0.7, label='Val Acc')
    axes[i, 1].set_title(f'{name} — Accuracy', fontweight='bold')
    axes[i, 1].set_xlabel('Epoch')
    axes[i, 1].set_ylabel('Accuracy')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_training_curves.png")

#validation comparison across all models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('All Models — Validation Comparison', fontsize=14, fontweight='bold')
for i, (name, h) in enumerate(histories.items()):
    d = h.history
    ep = range(1, len(d['loss']) + 1)
    label = name.split('(')[0].strip()
    ax1.plot(ep, d['val_loss'], color=COLORS[i], lw=2, label=label)
    ax2.plot(ep, d['val_accuracy'], color=COLORS[i], lw=2, label=label)
ax1.set_title('Validation Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02b_val_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02b_val_comparison.png")


#4. evaluation - confusion matrix, accuracy, model comparison
print("\n" + "=" * 70)
print("4.Evaluation on the test set")

results = {}
for name, model in trained.items():
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    results[name] = {'loss': loss, 'accuracy': acc}
    print(f"  {name}: loss={loss:.4f}, accuracy={acc:.4f} ({acc*100:.2f}%)")

#confusion matrices for each model
fig, axes = plt.subplots(1, len(MODELS), figsize=(7 * len(MODELS), 6))
fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
for i, (name, model) in enumerate(trained.items()):
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_title(f'{name}\nAcc: {results[name]["accuracy"]:.4f}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('03_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_confusion_matrices.png")

#bar chart comparing models with scoring thresholds
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
accs = [results[n]['accuracy'] * 100 for n in names]
bars = ax.bar(range(len(names)), accs, color=COLORS, edgecolor='black', width=0.6)
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f'{a:.2f}%', ha='center', fontweight='bold', fontsize=12)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace('(', '\n(') for n in names], fontsize=9)
ax.set_ylabel('Test Accuracy (%)')
ax.set_ylim(0, 105)
ax.set_title('Model Comparison — Test Accuracy', fontsize=14, fontweight='bold')
ax.axhline(80, color='green', ls='--', alpha=0.5, label='80%')
ax.axhline(70, color='orange', ls='--', alpha=0.5, label='70%')
ax.axhline(60, color='red', ls='--', alpha=0.5, label='60%')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('04_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_model_comparison.png")

#classification report for the best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = trained[best_name]
y_pred_best = best_model.predict(x_test, verbose=0).argmax(axis=1)
print(f"\n  Classification Report — {best_name}")
print(classification_report(y_test, y_pred_best, target_names=CLASS_NAMES))


#5. analysis - overfitting/underfitting & misclassifications
print("=" * 70)
print("5. Analysis")

#overfitting / underfitting evaluation
print("\nOverfitting / Underfitting\n")
for name, h in histories.items():
    d = h.history
    train_acc = d['accuracy'][-1]
    best_val = max(d['val_accuracy'])
    final_val = d['val_accuracy'][-1]
    gap = train_acc - final_val
    if gap > 0.15:
        verdict = "overfitting"
    elif best_val < 0.55:
        verdict = "underfitting"
    elif gap > 0.08:
        verdict = "slight overfitting"
    else:
        verdict = "good fit"
    print(f"  {name}")
    print(f"    Train acc: {train_acc:.4f} | Best val acc: {best_val:.4f} | "
          f"Gap: {gap:.4f} -> {verdict}\n")

#misclassification analysis
print("Misclassification Analysis\n")
cm_best = confusion_matrix(y_test, y_pred_best)

#top confused pairs
pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm_best[i][j] > 0:
            pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_best[i][j]))
pairs.sort(key=lambda x: x[2], reverse=True)

print("Top 10 confused pairs (True -> Predicted : count):")
for true_cls, pred_cls, count in pairs[:10]:
    print(f"    {true_cls:>12s} -> {pred_cls:<12s}: {count:4d}")

#per-class accuracy
print("\n Per-class accuracy:")
for i in range(10):
    total = cm_best[i].sum()
    correct = cm_best[i][i]
    print(f"    {CLASS_NAMES[i]:>12s}: {correct:4d}/{total:4d} = {correct/total:.4f}")

#misclassified examples visualization
misclassified = np.where(y_test != y_pred_best)[0]
print(f"\n  Total misclassified: {len(misclassified)}/{len(y_test)} "
      f"({len(misclassified)/len(y_test)*100:.1f}%)")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle(f'Misclassified Examples ({best_name})', fontsize=14, fontweight='bold')
np.random.seed(0)
sample_idx = np.random.choice(misclassified, size=min(10, len(misclassified)), replace=False)
for i, ax in enumerate(axes.flat):
    if i < len(sample_idx):
        ix = sample_idx[i]
        ax.imshow(x_test[ix])
        ax.set_title(f'True: {CLASS_NAMES[y_test[ix]]}\nPred: {CLASS_NAMES[y_pred_best[ix]]}',
                     fontsize=9, color='red')
    ax.axis('off')
plt.tight_layout()
plt.savefig('05_misclassified.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_misclassified.png")

#6. conclusions
print("\n" + "=" * 70)
print("6. Conclusion")

print("\n  Final Results:")
for n in results:
    a = results[n]['accuracy']
    print(f"    {n}: {a:.4f} ({a*100:.2f}%)")

best_acc = results[best_name]['accuracy']
print(f"\n  Best model: {best_name} — {best_acc:.4f} ({best_acc*100:.2f}%)")