import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Ayarlar
DATASET_PATH = 'dataset2'
CATEGORIES = ['glass_breaking', 'fall', 'silence', 'scream']
MIXED_PATH = 'mixed'

SR = 16000  # Sampling Rate
DURATION = 2  # 2 saniye
SAMPLES_PER_FILE = SR * DURATION
N_MELS = 128

# Mel-spectrogram çıkaran fonksiyon (kaydetme yok)
def extract_mel(file_path):
    audio, _ = librosa.load(file_path, sr=SR)
    if len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))
    else:
        audio = audio[:SAMPLES_PER_FILE]
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

# Dataset yükleme
def load_dataset():
    X = []
    y = []
    for idx, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)
        for file_name in os.listdir(category_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(category_path, file_name)
                mel_db = extract_mel(file_path)
                X.append(mel_db)
                y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

# CNN modeli
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((N_MELS, -1, 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Ana işlem
def main():
    print("Dataset yükleniyor...")
    X, y = load_dataset()

    # Eğitim ve doğrulama seti
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # CNN modelini oluştur
    model = create_model(X_train.shape[1:], len(CATEGORIES))
    model.summary()

    # Modeli eğit
    print("Model eğitiliyor...")
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

    # Mixed klasöründeki dosyaları test et
    print("\nMixed klasöründeki sesler tahmin ediliyor...")
    for file_name in os.listdir(MIXED_PATH):
        if file_name.endswith('.wav'):
            file_path = os.path.join(MIXED_PATH, file_name)
            mel_db = extract_mel(file_path)
            mel_db = np.expand_dims(mel_db, axis=0)  # Batch dimension
            prediction = model.predict(mel_db)
            predicted_class = np.argmax(prediction)
            print(f"{file_name} -> Tahmin: {CATEGORIES[predicted_class]}")

if __name__ == "__main__":
    main()
