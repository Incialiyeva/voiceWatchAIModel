import os
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configuration
DATASET_PATH = 'dataset2'
CATEGORIES    = ['glass_breaking', 'fall', 'silence', 'scream']
MIXED_PATH    = 'mixed'

SR             = 16000  # Sampling rate (Hz)
DURATION       = 2      # Duration of each audio clip in seconds
SAMPLES_PER_FILE = SR * DURATION
N_MELS         = 128    # Number of Mel bands

# Function to extract mel-spectrogram (no saving to file)
def extract_mel(file_path):
    audio, _ = librosa.load(file_path, sr=SR)
    if len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))
    else:
        audio = audio[:SAMPLES_PER_FILE]
    mel    = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# Load dataset and labels from folders
def load_dataset():
    X, y = [], []
    for idx, category in enumerate(CATEGORIES):
        category_dir = os.path.join(DATASET_PATH, category)
        for fname in os.listdir(category_dir):
            if not fname.endswith('.wav'):
                continue
            path = os.path.join(category_dir, fname)
            mel  = extract_mel(path)
            X.append(mel)
            y.append(idx)
    return np.array(X), np.array(y)

# Build a simple CNN model
def create_model(input_shape, num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((N_MELS, -1, 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='audio_classifier')

def main():
    print("Loading dataset…")
    X, y = load_dataset()

    # train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # create & compile
    model = create_model(X_train.shape[1:], len(CATEGORIES))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # train
    print("\nTraining model…")
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_val, y_val)
    )

    # —————————————————————————————————————————
    # SavedModel olarak dışa aktarma (Keras 3)
    saved_model_dir = "saved_model"
    model.export(saved_model_dir)
    print(f"\n✅ SavedModel formatında kaydedildi: {saved_model_dir}/")
    # —————————————————————————————————————————

    # karışık (mixed) klasöründen 3 rasgele dosya alıp tahmin et
    print("\nPredicting 3 random audio files from 'mixed' folder…")
    all_wavs = [f for f in os.listdir(MIXED_PATH) if f.endswith('.wav')]
    picks    = random.sample(all_wavs, min(3, len(all_wavs)))

    for fname in picks:
        fpath = os.path.join(MIXED_PATH, fname)
        mel   = extract_mel(fpath)
        batch = np.expand_dims(mel, axis=0)
        preds = model.predict(batch)
        cls   = np.argmax(preds)

        print(f"{fname} → Predicted: {CATEGORIES[cls]}")

        # Görselleştir
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel,
                                 sr=SR,
                                 hop_length=512,
                                 x_axis='time',
                                 y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram - {fname}\nPredicted: {CATEGORIES[cls]}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
