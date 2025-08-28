### Updated 2025.08.16 for TF2
# === Updated for TensorFlow 2.x / Keras (2025-08-16) ===
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Optional: limit visible GPUs BEFORE TensorFlow initializes
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Data utils (kept if you still need ImageDataGenerator)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Layers
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Lambda, Dropout, Activation, LSTM, GRU,
    TimeDistributed, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D,
    BatchNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D,
    concatenate, ZeroPadding2D, Reshape, GlobalAveragePooling2D,
    GlobalMaxPooling2D, AveragePooling2D, ELU
)

# Optimizers
from tensorflow.keras.optimizers import Adam, RMSprop

# Callbacks
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, ModelCheckpoint, EarlyStopping,
    CSVLogger, TensorBoard
)

# Backend / models
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model

# Sklearn
from sklearn.model_selection import train_test_split

# If you use skimage.resize below
from skimage.transform import resize

# ---------- GPU memory growth (TF2 way) ----------
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass  # Ignore if not supported

# ---------- Model blocks ----------
def base_conv_block(num_conv_filters, kernel_size):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        out = Conv2D(num_conv_filters, kernel_size, padding='same')(x)
        return out
    return f

def multi_scale_block(num_conv_filters):
    def f(input_):
        branch1x1 = base_conv_block(num_conv_filters, 1)(input_)

        branch3x3 = base_conv_block(num_conv_filters, 1)(input_)
        branch3x3 = base_conv_block(num_conv_filters, 3)(branch3x3)

        branch5x5 = base_conv_block(num_conv_filters, 1)(input_)
        branch5x5 = base_conv_block(num_conv_filters, 5)(branch5x5)

        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(input_)
        branchpool = base_conv_block(num_conv_filters, 1)(branchpool)

        out = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=-1)
        return out
    return f

def dense_block(num_dense_blocks, num_conv_filters):
    def f(input_):
        x = input_
        for _ in range(num_dense_blocks):
            out = multi_scale_block(num_conv_filters)(x)
            x = concatenate([x, out], axis=-1)
        return x
    return f

def transition_block(num_conv_filters):
    def f(input_):
        x = BatchNormalization()(input_)
        x = Activation('relu')(x)
        x = Conv2D(num_conv_filters, 1)(x)
        out = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return out
    return f

def multi_scale_level_cnn(input_shape, num_dense_blocks, num_conv_filters, num_classes):
    model_input = Input(shape=input_shape)

    x = Conv2D(num_conv_filters, 3, padding='same')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    x = dense_block(num_dense_blocks, num_conv_filters)(x)
    x = transition_block(num_conv_filters)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    model_output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model_input, outputs=model_output)
    return model

# ---------- Data helpers ----------
def process_data_for_conv2D(X, resize_shape=None):
    X_conv2D = []
    for sample in X:
        sample = np.reshape(sample, newshape=(sample.shape[0], sample.shape[1], 1))
        if resize_shape:
            sample = resize(sample, output_shape=(*resize_shape, 1), preserve_range=True, anti_aliasing=True)
        X_conv2D.append(sample.astype(np.float32))
    return np.array(X_conv2D, dtype=np.float32)

def data_iter(X, y, batch_size):
    num_samples = X.shape[0]
    idx = np.arange(num_samples)
    while True:
        for i in range(0, num_samples, batch_size):
            j = idx[i:min(i+batch_size, num_samples)]
            yield X[j, :], y[j, :]

def train_val_test_split(X, y, train_size, val_size, test_size, random_state=42):
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test,
        test_size=test_size/(test_size + val_size),
        stratify=y_val_test,
        random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

import tensorflow as tf

def spec_augment_tf(mel_spectrogram,
                    freq_mask_param=15,
                    time_mask_param=20,
                    num_freq_masks=1,
                    num_time_masks=1):
    """
    mel_spectrogram: tf.Tensor of shape (freq_bins, time_steps, channels) or (freq_bins, time_steps)
    Returns augmented spectrogram of the same shape
    """
    augmented = tf.identity(mel_spectrogram)

    # Ensure shape is (freq, time, channels)
    if len(augmented.shape) == 2:
        augmented = tf.expand_dims(augmented, axis=-1)

    freq_bins = tf.shape(augmented)[0]
    time_steps = tf.shape(augmented)[1]

    # Frequency masking
    for _ in range(num_freq_masks):
        f = tf.random.uniform([], minval=0, maxval=freq_mask_param + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, freq_bins - f), dtype=tf.int32)
        mask = tf.concat([
            tf.ones((f0, time_steps, 1), dtype=augmented.dtype),
            tf.zeros((f, time_steps, 1), dtype=augmented.dtype),
            tf.ones((freq_bins - f0 - f, time_steps, 1), dtype=augmented.dtype)
        ], axis=0)
        augmented *= mask

    # Time masking
    for _ in range(num_time_masks):
        t = tf.random.uniform([], minval=0, maxval=time_mask_param + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tf.maximum(1, time_steps - t), dtype=tf.int32)
        mask = tf.concat([
            tf.ones((freq_bins, t0, 1), dtype=augmented.dtype),
            tf.zeros((freq_bins, t, 1), dtype=augmented.dtype),
            tf.ones((freq_bins, time_steps - t0 - t, 1), dtype=augmented.dtype)
        ], axis=1)
        augmented *= mask

    return augmented


# Apply SpecAugment on-the-fly
def augment_fn(x, y):
    x_aug = spec_augment_tf(x)
    return x_aug, y




# ---------- Dataset prep ----------
mask = ((df5['melSpec'].notna()) & (df5['timbre'].notna()) & (df5['yearFloor'] == 0))
df5[mask].groupby('augmented')['id'].count()
# augmented
# False    190
# True     380


df0 = df5[mask].groupby(['pitchShift','id']).first().reset_index()
# df0 = df0[(df0['Jahr'].astype('int') >= 2008) & (df0['Jahr'].astype('int') <= 2018)] #Let's use full dataset
df0.groupby('augmented')['id'].count()
# augmented
# False    189
# True     380

X_melspec = df0['melSpec']
y = df0['timbre']            # ensure this is one-hot if using categorical_crossentropy

X_melspec = process_data_for_conv2D(X_melspec)
print(X_melspec.shape)
print(y.shape)
#Need to transpose 
def transposeMelSpec(x):
    return x.T
# df0['melSpecHalf'] = df0['melSpec'].apply(lambda m: m[:, m.shape[1]//2 :])
# df0['melSpecT'] = df0['melSpecHalf'].apply(lambda x: transposeMelSpec(x))
df0['melSpecT'] = df0['melSpec'].apply(lambda x: transposeMelSpec(x))
X_melspec = df0['melSpecT']
y = df0['timbre']            # ensure this is one-hot if using categorical_crossentropy
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Suppose y is your Series of strings ("A"/"B" or "mittel/dunkel"/"hell")
encoder = LabelEncoder()
y_int = encoder.fit_transform(y)   # Now y_int is 0/1 instead of strings

# One-hot encode
y_onehot = to_categorical(y_int, num_classes=2)
X_melspec = process_data_for_conv2D(X_melspec)
print(X_melspec.shape)
print(y_onehot.shape)
# (214, 1182, 128, 1)
# (214,)


# ---------- Build / summarize ----------
num_classes = 2#10
model = multi_scale_level_cnn(
    input_shape=(X_melspec.shape[1], X_melspec.shape[2], X_melspec.shape[3]),
    num_dense_blocks=3, num_conv_filters=32, num_classes=num_classes
)
model.summary()

# ---------- Train / eval (k-fold style) ----------
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold


### Check optimal convolutional filter number

filters_to_test = [8, 16, 32]
epochs_test = 2
folds_test = 1  # just first fold to save time

for num_filters in filters_to_test:
    print(f"\n=== Testing num_conv_filters={num_filters} ===")
    
    # Recreate model for each run
    model = multi_scale_level_cnn(
        input_shape=(X_melspec.shape[1], X_melspec.shape[2], X_melspec.shape[3]),
        num_dense_blocks=3,
        num_conv_filters=num_filters,
        num_classes=num_classes
    )
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    
    # Train on just first fold
    X_train, X_val, y_train, y_val = train_test_split(
        X_melspec, y_onehot, test_size=0.1, stratify=y_onehot
    )
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_test,
        batch_size=batch_size,
        verbose=1
    )
    end_time = time.time()
    
    print(f"Training time: {end_time - start_time:.1f} sec")
    print(f"Final val_accuracy: {history.history['val_accuracy'][-1]:.3f}")

# === Testing num_conv_filters=8 ===
# Epoch 1/2
# 64/64 ━━━━━━━━━━━━━━━━━━━━ 94s 1s/step - accuracy: 0.5053 - loss: 0.7362 - val_accuracy: 0.4386 - val_loss: 0.7272
# Epoch 2/2
# 64/64 ━━━━━━━━━━━━━━━━━━━━ 70s 1s/step - accuracy: 0.6615 - loss: 0.6425 - val_accuracy: 0.5965 - val_loss: 0.6401
# Training time: 164.4 sec
# Final val_accuracy: 0.596

# === Testing num_conv_filters=16 ===
# Epoch 1/2
# 64/64 ━━━━━━━━━━━━━━━━━━━━ 173s 2s/step - accuracy: 0.5550 - loss: 0.7121 - val_accuracy: 0.5614 - val_loss: 13.1761
# Epoch 2/2
# 64/64 ━━━━━━━━━━━━━━━━━━━━ 146s 2s/step - accuracy: 0.5869 - loss: 0.6641 - val_accuracy: 0.5614 - val_loss: 3.8527
# Training time: 318.6 sec
# Final val_accuracy: 0.561

# === Testing num_conv_filters=32 ===
# Epoch 1/2
# 64/64 ━━━━━━━━━━━━━━━━━━━━ 616s 9s/step - accuracy: 0.5477 - loss: 0.6985 - val_accuracy: 0.5614 - val_loss: 3.5203
# Epoch 2/2
# 64/64 ━━━━━━━━━━━━━━━━━━━━ 475s 7s/step - accuracy: 0.6380 - loss: 0.6477 - val_accuracy: 0.5614 - val_loss: 2.1014
# Training time: 1092.0 sec
# Final val_accuracy: 0.561

# --- Hyperparameters ---
epochs = 10#100
batch_size = 8
learning_rate = 0.01
num_classes = 2  # binary classification (hell vs mittel/dunkel)

# Define KFold splitter
k_fold = 5  # or 10
num_conv_filters_default = 8 #32
kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)

train_loss_record, train_acc_record = [], []
val_loss_record, val_acc_record = [], []
test_loss_record, test_acc_record = [], []

###Only change from Liu et al. is implementing specAugment
for fold, (train_val_idx, test_idx) in enumerate(kf.split(X_melspec, y)):
    ...

    print(f"\n=== Fold {fold+1}/{k_fold} ===")

    # Split into train+val and test
    X_train_val, X_test = X_melspec[train_val_idx], X_melspec[test_idx]
    y_train_val, y_test = y_onehot[train_val_idx], y_onehot[test_idx]

    # Further split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, stratify=y_train_val
    )
    
    ###This is the SpecAugment implementation
    # --- Convert training data into tf.data.Dataset ---
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    train_dataset = (
        train_dataset
        .shuffle(buffer_size=len(X_train))
        .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    

    # Validation dataset (no augmentation)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    ###SpecAugment complete

    # --- File paths for saving ---
    file_name = f"fold_{fold+1}_best_model.keras"
    csv_path = f"fold_{fold+1}_training_log.csv"

    # --- Callbacks ---
    lr_change = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(
        file_name, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=False, verbose=1
    )
    csv_logger = CSVLogger(csv_path)
    callbacks = [lr_change, model_checkpoint, early_stopping, csv_logger]

    # --- Model ---
    opt = Adam(learning_rate=learning_rate)
    model = multi_scale_level_cnn(
        input_shape=(X_melspec.shape[1], X_melspec.shape[2], X_melspec.shape[3]),
        num_dense_blocks=3,
        num_conv_filters=num_conv_filters_default,#32,
        num_classes=num_classes,
    )
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)

    # --- Train ---
    model.fit(
        #X_train, y_train, #Liu et al.
        train_dataset, #SpecAugment
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_dataset,#(X_val, y_val), #Liu et al.
        verbose=1,
        callbacks=callbacks
    )

    # --- Reload best model ---
    model_best = load_model(file_name)

    # --- Evaluate ---
    ###Changed to train_, val_, and test_dataset from Liu et al.
    train_loss, train_acc = model_best.evaluate(train_dataset, verbose=0)#(X_train, y_train, batch_size=batch_size, verbose=0)
    val_loss, val_acc     = model_best.evaluate(val_dataset, verbose=0)#(X_val, y_val, batch_size=batch_size, verbose=0)
    test_loss, test_acc   = model_best.evaluate(test_dataset, verbose=0)#(X_test, y_test, batch_size=batch_size, verbose=0)

    # Save results
    train_loss_record.append(train_loss);  train_acc_record.append(train_acc)
    val_loss_record.append(val_loss);      val_acc_record.append(val_acc)
    test_loss_record.append(test_loss);    test_acc_record.append(test_acc)

    print(f"Fold {fold+1} | Train acc {train_acc:.3f}, Val acc {val_acc:.3f}, Test acc {test_acc:.3f}")

###Visualizations

### Plot Global Confusion Matrix normalized across five folds:
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# Initialize global confusion matrix
global_cm = np.zeros((num_classes, num_classes), dtype=int)

for fold, (train_val_idx, test_idx) in enumerate(kf.split(X_melspec, y)):
    print(f"\n=== Evaluating Fold {fold+1}/{k_fold} ===")

    # Split test set for this fold
    X_test = X_melspec[test_idx]
    y_test = y_onehot[test_idx]
    y_test_labels = np.argmax(y_test, axis=1)  # convert one-hot to class indices

    # Reload the best model for this fold
    model_path = f"fold_{fold+1}_best_model.keras"
    model = load_model(model_path)

    # Predict
    y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    # Compute confusion matrix for this fold
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=range(num_classes))

    # Add to global confusion matrix
    global_cm += cm

# Normalize if you want per-class proportions
global_cm_normalized = global_cm.astype("float") / global_cm.sum(axis=1)[:, np.newaxis]

print("\n=== Global Confusion Matrix (counts) ===")
print(global_cm)

print("\n=== Global Confusion Matrix (normalized) ===")
print(np.round(global_cm_normalized, 3))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class_names = ['hell', 'mittel/dunkel']

cm = global_cm
normalize = False
plt.figure(figsize=(6,5))
ax = sns.heatmap(cm, annot=False, cmap="Blues", 
                 xticklabels=class_names, yticklabels=class_names,
                 cbar=True, square=True)

# Draw numbers manually
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text_val = f"{int(cm[i,j])}" if not normalize else f"{cm[i,j]:.2f}"
        ax.text(j + 0.5, i + 0.5, text_val, ha='center', va='center', color='black', fontsize=14)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
# plt.show()
plt.savefig('ConfusionMatrix.png')

normalize = True
plt.figure(figsize=(6,5))
if normalize:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm = cm / np.maximum(row_sums, 1)  # prevent division by zero
ax = sns.heatmap(cm, annot=False, cmap="Blues", 
                 xticklabels=class_names, yticklabels=class_names,
                 cbar=True, square=True)

# Draw numbers manually
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text_val = f"{int(cm[i,j])}" if not normalize else f"{cm[i,j]:.2f}"
        ax.text(j + 0.5, i + 0.5, text_val, ha='center', va='center', color='black', fontsize=14)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('ConfusionMatrixNorm.png')
# plt.show()


t1 = pd.read_csv('fold_1_training_log.csv')
t1['Log'] = 1
mask1 = t1['val_accuracy'] < t1['val_accuracy'].max()
t1 = t1.drop(t1[mask1].index)
t2 = pd.read_csv('fold_2_training_log.csv')
t2['Log'] = 2
mask2 = t2['val_accuracy'] < t2['val_accuracy'].max()
t2 = t2.drop(t2[mask2].index)
t3 = pd.read_csv('fold_3_training_log.csv')
t3['Log'] = 3
mask3 = t3['val_accuracy'] < t3['val_accuracy'].max()
t3 = t3.drop(t3[mask3].index)
t4 = pd.read_csv('fold_4_training_log.csv')
t4['Log'] = 4
mask4 = t4['val_accuracy'] < t4['val_accuracy'].max()
t4 = t4.drop(t4[mask4].index)
t5 = pd.read_csv('fold_5_training_log.csv')
t5['Log'] = 5
mask5 = t5['val_accuracy'] < t5['val_accuracy'].max()
t5 = t5.drop(t5[mask5].index)
tL = pd.concat([t1,t2,t3,t4,t5])
tL.describe()
       # accuracy   loss  val_accuracy  val_loss
# count     5.000  5.000         5.000     5.000
# mean      0.648  0.640         0.689     0.640
# std       0.013  0.018         0.089     0.065
# min       0.627  0.614         0.578     0.558
# 25%       0.644  0.636         0.644     0.582
# 50%       0.654  0.636         0.667     0.682
# 75%       0.657  0.654         0.756     0.682
# max       0.660  0.660         0.800     0.696

import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load the model from epoch with median accuracy
# -----------------------------
model_path = "fold_4_best_model.keras"  # Make sure you saved this during training
model = load_model(model_path)
model.summary()
model.trainable = False  # Freeze layers to avoid accidental training
for i, layer in enumerate(model.layers):
    try:
        print(i, layer.name, layer.output.shape)
    except AttributeError:
        print(i, layer.name, "no output shape")

# 0 input_layer_4 (None, 1182, 128, 1)
# 1 conv2d_80 (None, 1182, 128, 8)
# 2 batch_normalization_84 (None, 1182, 128, 8)
# 3 activation_84 (None, 1182, 128, 8)
# 4 max_pooling2d_16 (None, 295, 128, 8)
# 5 batch_normalization_86 (None, 295, 128, 8)
# 6 batch_normalization_88 (None, 295, 128, 8)
# 7 activation_86 (None, 295, 128, 8)
# 8 activation_88 (None, 295, 128, 8)
# 9 conv2d_82 (None, 295, 128, 8)
# 10 conv2d_84 (None, 295, 128, 8)
# 11 max_pooling2d_17 (None, 295, 128, 8)
# 12 batch_normalization_85 (None, 295, 128, 8)
# 13 batch_normalization_87 (None, 295, 128, 8)
# 14 batch_normalization_89 (None, 295, 128, 8)
# 15 batch_normalization_90 (None, 295, 128, 8)
# 16 activation_85 (None, 295, 128, 8)
# 17 activation_87 (None, 295, 128, 8)
# 18 activation_89 (None, 295, 128, 8)
# 19 activation_90 (None, 295, 128, 8)
# 20 conv2d_81 (None, 295, 128, 8)
# 21 conv2d_83 (None, 295, 128, 8)
# 22 conv2d_85 (None, 295, 128, 8)
# 23 conv2d_86 (None, 295, 128, 8)
# 24 concatenate_24 (None, 295, 128, 32)
# 25 concatenate_25 (None, 295, 128, 40)
# 26 batch_normalization_92 (None, 295, 128, 40)
# 27 batch_normalization_94 (None, 295, 128, 40)
# 28 activation_92 (None, 295, 128, 40)
# 29 activation_94 (None, 295, 128, 40)
# 30 conv2d_88 (None, 295, 128, 8)
# 31 conv2d_90 (None, 295, 128, 8)
# 32 max_pooling2d_18 (None, 295, 128, 40)
# 33 batch_normalization_91 (None, 295, 128, 40)
# 34 batch_normalization_93 (None, 295, 128, 8)
# 35 batch_normalization_95 (None, 295, 128, 8)
# 36 batch_normalization_96 (None, 295, 128, 40)
# 37 activation_91 (None, 295, 128, 40)
# 38 activation_93 (None, 295, 128, 8)
# 39 activation_95 (None, 295, 128, 8)
# 40 activation_96 (None, 295, 128, 40)
# 41 conv2d_87 (None, 295, 128, 8)
# 42 conv2d_89 (None, 295, 128, 8)
# 43 conv2d_91 (None, 295, 128, 8)
# 44 conv2d_92 (None, 295, 128, 8)
# 45 concatenate_26 (None, 295, 128, 32)
# 46 concatenate_27 (None, 295, 128, 72)
# 47 batch_normalization_98 (None, 295, 128, 72)
# 48 batch_normalization_100 (None, 295, 128, 72)
# 49 activation_98 (None, 295, 128, 72)
# 50 activation_100 (None, 295, 128, 72)
# 51 conv2d_94 (None, 295, 128, 8)
# 52 conv2d_96 (None, 295, 128, 8)
# 53 max_pooling2d_19 (None, 295, 128, 72)
# 54 batch_normalization_97 (None, 295, 128, 72)
# 55 batch_normalization_99 (None, 295, 128, 8)
# 56 batch_normalization_101 (None, 295, 128, 8)
# 57 batch_normalization_102 (None, 295, 128, 72)
# 58 activation_97 (None, 295, 128, 72)
# 59 activation_99 (None, 295, 128, 8)
# 60 activation_101 (None, 295, 128, 8)
# 61 activation_102 (None, 295, 128, 72)
# 62 conv2d_93 (None, 295, 128, 8)
# 63 conv2d_95 (None, 295, 128, 8)
# 64 conv2d_97 (None, 295, 128, 8)
# 65 conv2d_98 (None, 295, 128, 8)
# 66 concatenate_28 (None, 295, 128, 32)
# 67 concatenate_29 (None, 295, 128, 104)
# 68 batch_normalization_103 (None, 295, 128, 104)
# 69 activation_103 (None, 295, 128, 104)
# 70 conv2d_99 (None, 295, 128, 8)
# 71 average_pooling2d_4 (None, 147, 64, 8)
# 72 batch_normalization_104 (None, 147, 64, 8)
# 73 activation_104 (None, 147, 64, 8)
# 74 global_average_pooling2d_4 (None, 8)
# 75 dense_4 (None, 2)

target_layer = 'conv2d_99'
# -----------------------------
# Step 2: Prepare your data
# -----------------------------
# X_background: small subset of training data for SHAP
# X_test: data you want explanations for

import pandas as pd
import numpy as np
import shap

# Suppose df is your full dataset with 'timbre' and 'Stimmfach'
df0['stratify_key'] = df0['timbre'].astype(str) + "_" + df0['Stimmfach'].astype(str)

def stratified_sample(df, n_samples):
    # Calculate how many samples per group (proportional)
    group_counts = df['stratify_key'].value_counts(normalize=True)
    
    sampled_indices = []
    for group, prop in group_counts.items():
        group_df = df[df['stratify_key'] == group]
        n_group_samples = max(1, int(round(prop * n_samples)))  # at least 1 per group
        sampled_indices.extend(np.random.choice(group_df.index, n_group_samples, replace=False))
    
    return df.loc[sampled_indices]





# mask = df0['augmented'] == False
soloMask = ((df0['SoloChor'] == 'Solo') & (df0['augmented'] == False))
# Example: 120 background samples
X_background_df = stratified_sample(df0[soloMask], 15)
X_background = np.stack(X_background_df['melSpecT'].values)

sicherMask = ((df0['zulassung.hno'] == 'sicher, Solo') & (df0['augmented'] == False))
# Example: 30 test samples
X_test_df = stratified_sample(df0[sicherMask], 4)
X_test = np.stack(X_test_df['melSpecT'].values)


def grad_cam(model, img, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(img, 0))
        target_class = tf.argmax(predictions[0])
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
def grad_cam_single(model, img, layer_name):
    """
    Compute Grad-CAM heatmap for a single input sample.

    Args:
        model: Trained tf.keras model
        img: Single input sample (H, W) or (H, W, 1)
        layer_name: Name of target convolutional layer

    Returns:
        heatmap: 2D numpy array (H, W) normalized between 0 and 1
    """

    # Ensure input is (1, H, W, 1)
    img = np.array(img)
    if img.ndim == 2:               # (H, W)
        img = img[np.newaxis, :, :, np.newaxis]
    elif img.ndim == 3:
        if img.shape[-1] != 1:
            img = img[np.newaxis, :, :, :]
        else:
            img = img[np.newaxis, :, :, :]  # (1, H, W, 1)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        
        # If predictions is a list (multi-output), pick the first output
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        
        target_class = tf.argmax(predictions[0])
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-10)
    return heatmap#.numpy()

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return heatmap.numpy()

import tensorflow as tf
import numpy as np



import librosa
import numpy as np
import matplotlib.pyplot as plt
# Example usage:
for i in range(len(X_test)):


    # Parameters
    n_mels = 128
    fmin = 1000     # minimum frequency
    fmax = 5000    # maximum frequency in Hz (depends on your audio)

    # Compute approximate linear frequencies for each Mel bin
    mel_bins = np.arange(n_mels)
    frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Example plotting
    spec = X_test[i].squeeze()  # original Mel spectrogram
    heatmap = grad_cam_single(model, spec, target_layer)
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    threshold = 0.6
    heatmap_highlight = np.where(heatmap_norm >= threshold, heatmap_norm, 0)

    plt.figure(figsize=(10, 4))
    plt.imshow(spec.T, aspect='auto', origin='lower', cmap='gray', 
               extent=[0, spec.shape[0], frequencies[0], frequencies[-1]])

    plt.imshow(heatmap_highlight.T, aspect='auto', origin='lower', cmap='jet', 
               alpha=0.7, extent=[0, spec.shape[0], frequencies[0], frequencies[-1]])

    plt.xlabel('Time frames')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram with Grad-CAM features highlighted')
    plt.colorbar(label='Grad-CAM intensity')
    plt.show()


### Full heatmap investigation:
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm


# 1. Calculate normalized heatmaps
df_0 = df0[df0['augmented'] == False][['id','date','melSpecT','timbre']]
df_0['heatmap'] = df_0['melSpecT'].apply(lambda x: grad_cam_single(model, x, target_layer))

# Oops, re-add the labels from df0 using 'id' as the key
# df_0 = df_0.merge(df0[['id','timbre']], on='id', how='left')
# ----------------------------
# 2. Group by class
# ----------------------------

heatmaps_norm = df_0['heatmap'].values  # array of 2D arrays
labels = df_0['timbre'].values
classes = np.unique(labels)

mean_maps = {}
median_maps = {}
std_maps = {}

for c in classes:
    # Select indices for this class
    idx = np.where(labels == c)[0]
    # Stack the heatmaps along a new axis
    stacked = np.stack(heatmaps_norm[idx], axis=0)  # shape: (n_samples, H, W)
    # Compute statistics
    mean_maps[c] = np.mean(stacked, axis=0)
    median_maps[c] = np.median(stacked, axis=0)
    std_maps[c] = np.std(stacked, axis=0)


# ----------------------------
# 3. Collapse to freq/time importance
# ----------------------------
freq_importance = {c: mean_maps[c].mean(axis=0) for c in classes}  # (F,)
time_importance = {c: mean_maps[c].mean(axis=1) for c in classes}  # (T,)

# ----------------------------
# 4. Statistical comparison (two-class only)
# ----------------------------

if len(classes) == 2:
    c0, c1 = classes

    # Get indices for each class
    idx0 = np.where(labels == c0)[0]
    idx1 = np.where(labels == c1)[0]

    # Stack into 3D arrays (n_samples, T, F)
    stacked0 = np.stack(heatmaps_norm[idx0], axis=0)
    stacked1 = np.stack(heatmaps_norm[idx1], axis=0)

    # Average over time (axis=1) to get per-frequency values
    A = stacked0.mean(axis=1)  # shape: (n0, F)
    B = stacked1.mean(axis=1)  # shape: (n1, F)

    # t-test per frequency
    pvals = np.array([ttest_ind(A[:,f], B[:,f], equal_var=False).pvalue
                      for f in range(A.shape[1])])

    # FDR correction
    reject, pvals_fdr = smm.fdrcorrection(pvals, alpha=0.05)
    significant_bins = np.where(reject)[0]
    print(f"Significant frequency bins (FDR corrected): {significant_bins}")

#No significant frequency bins

###Same process for the 2D grid:
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smm

# ----------------------------
# 1. Organize into arrays
# ----------------------------
classes = np.unique(labels)
assert len(classes) == 2, "Only supports two-class comparison"

c0, c1 = classes
idx0 = np.where(labels == c0)[0]
idx1 = np.where(labels == c1)[0]

# Stack into 3D arrays: (n_samples, T, F)
stacked0 = np.stack(heatmaps_norm[idx0], axis=0)  # shape (n0, T, F)
stacked1 = np.stack(heatmaps_norm[idx1], axis=0)  # shape (n1, T, F)

n0, T, F = stacked0.shape
n1, _, _ = stacked1.shape

# ----------------------------
# 2. T-test at each time–frequency bin
# ----------------------------
pvals = np.zeros((T, F))

for t in range(T):
    for f in range(F):
        pvals[t, f] = ttest_ind(
            stacked0[:, t, f],
            stacked1[:, t, f],
            equal_var=False
        ).pvalue

# ----------------------------
# 3. FDR correction across all bins
# ----------------------------
pvals_flat = pvals.ravel()
reject, pvals_fdr = smm.fdrcorrection(pvals_flat, alpha=0.05)

# Reshape back to 2D (time × frequency)
reject_2d = reject.reshape(T, F)
pvals_fdr_2d = pvals_fdr.reshape(T, F)

# ----------------------------
# 4. Report results
# ----------------------------
significant_bins = np.argwhere(reject_2d)
print(f"Number of significant time–frequency bins: {len(significant_bins)}")
print(f"Example significant bins (t, f): {significant_bins[:10]}")
### No individual bins significant

# ----------------------------
# 5. Plotting utilities
# ----------------------------
def plot_heatmap(mat, title='heatmap', xlabel='Time frames', ylabel='Frequency bins'):
    plt.figure(figsize=(6,4))
    plt.imshow(mat.T, origin='lower', aspect='auto', cmap='hot')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.colorbar(label='Normalized attribution')
    plt.show()


def get_mel(y, sr, n_mels=128,f_min=1000,f_max=5000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=f_max, fmin=f_min)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

### Plot time and frequency importances independently

ref_path = 'C:/Users/Reuben/Documents/Code/Promotionsvorhaben/Sandbox/0011&2009_01_27&test5.wav' # sicher, Solo soprano with 0.06 s difference from median duration
ref_y, sr = librosa.load(ref_path, sr=16000)
ref_mel = get_mel(ref_y, sr)

spec = ref_mel
#Let's plot over the reference spectrogram:
n_mels = spec.shape[0]   # number of mel bins (128 in your case)
fmin, fmax = 1000, 5000  # same as used in get_mel()
mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
plt.close('all')
# Make sure you have sr and hop_length defined from your mel calculation
# sr = sr
hop_length = 512

n_frames = spec.shape[1]
times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

# === Time Importance ===
fig, ax1 = plt.subplots(figsize=(12,4))

# Base spectrogram
im = ax1.imshow(spec,
                aspect='auto',
                origin='lower',
                cmap='gray',
                extent=[times[0], times[-1], mel_freqs[0], mel_freqs[-1]])
fig.colorbar(im, ax=ax1, label="Mel magnitude")

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("Time importance over spectrogram")

# Secondary y-axis for importance (0–1)
ax2 = ax1.twinx()
ax2.set_ylim(0,1)
ax2.set_ylabel("Time Importance")

# Plot importance curves
for c in classes:
    ax2.plot(np.linspace(times[0], times[-1], len(time_importance[c])),
             time_importance[c],
             label=str(c), linewidth=2)

ax2.legend(loc="upper right")
plt.tight_layout()
# plt.show()
plt.savefig("TimeImportance.png")

plt.close('all')
# === Frequency Importance ===
fig, ax1 = plt.subplots(figsize=(12,4))

# Base spectrogram
im = ax1.imshow(spec,
                aspect='auto',
                origin='lower',
                cmap='gray',
                extent=[times[0], times[-1], mel_freqs[0], mel_freqs[-1]])
fig.colorbar(im, ax=ax1, label="Mel magnitude")

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_title("Frequency importance over spectrogram")

# Secondary x-axis for importance (0–1)
ax2 = ax1.twiny()
ax2.set_xlim(0,1)
ax2.set_xlabel("Frequency Importance")

# Plot importance curves
for c in classes:
    imp = freq_importance[c]
    imp = (imp - imp.min()) / (imp.max() - imp.min())  # normalize 0–1
    ax2.plot(imp, mel_freqs, label=str(c), linewidth=2)

ax2.legend(loc="upper right")
plt.tight_layout()
# plt.show()
plt.savefig("FrequencyImportance.png")


import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

###Delta Attributions GradCam

# Load reference spectrogram (mel)
ref_path = 'C:/Users/Reuben/Documents/Code/Promotionsvorhaben/Sandbox/0011&2009_01_27&test5.wav'
ref_y, sr = librosa.load(ref_path, sr=16000)
ref_mel = get_mel(ref_y, sr)

# Parameters
hop_length = 512  # must match get_mel
time_len = diff_map.shape[0]
freq_len = diff_map.shape[1]

# Frequency axis in Hz
fmin, fmax = 1000, 5000
mel_freqs = librosa.mel_frequencies(n_mels=freq_len, fmin=fmin, fmax=fmax)

# Time axis in seconds
n_frames = ref_mel.shape[1]
times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

# Plot reference spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(ref_mel,
           aspect='auto',
           origin='lower',
           cmap='gray',
           extent=[times[0], times[-1], mel_freqs[0], mel_freqs[-1]])

# Overlay difference heatmap (rescale if time dimensions differ)
if ref_mel.shape[1] != time_len:
    diff_resized = np.resize(diff_map, (time_len, freq_len))  # crude resize
else:
    diff_resized = diff_map

plt.imshow(diff_resized.T,
           aspect='auto',
           origin='lower',
           extent=[times[0], times[-1], mel_freqs[0], mel_freqs[-1]],
           cmap='bwr',
           alpha=0.3)

# Custom legend patches
red_patch = mpatches.Patch(color='red', label=f'Higher attribution for {c0}')
blue_patch = mpatches.Patch(color='blue', label=f'Higher attribution for {c1}')
plt.legend(handles=[red_patch, blue_patch],
           loc='lower center',
           bbox_to_anchor=(0.5, 0.98),
           ncol=2,
           frameon=False)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title(f"Difference heatmap over spectrogram ({c0} - {c1})\n")
plt.colorbar(label="Δ attribution")
plt.tight_layout()
# plt.show()
plt.savefig("DeltaAttributions.png")