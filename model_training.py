import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization


# 📌 Ses Dosyasını Yükleme
def load_audio(file_path, sr=16000):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Hata - Dosya yüklenemedi: {file_path}, Hata: {e}")
        return None, None

# 📌 MFCC Öznitelikleri Çıkarma
def extract_mfcc(file_path, n_mfcc=40):
    audio, sr = load_audio(file_path)
    if audio is None:
        return None
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"MFCC çıkarma hatası - {file_path}, Hata: {e}")
        return None

# 📌 Dataset'i İşleme
def process_dataset(dataset_path, categories):
    all_features, all_labels = [], []
    for category in categories:
        folder = os.path.join(dataset_path, category)
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.endswith('.wav'):
                    file_path = os.path.join(folder, filename)
                    feature_vector = extract_mfcc(file_path)
                    if feature_vector is not None:
                        all_features.append(feature_vector)
                        all_labels.append(category)
    return all_features, all_labels

# 🔥 Eğitim İçin Ayarlar
dataset_path = 'dataset2'
categories = ['glass_breaking', 'fall', 'silence', 'scream']
all_features, all_labels = process_dataset(dataset_path, categories)

X = np.array(all_features)
y = np.array(all_labels)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

X = X.reshape(-1, 40, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded)

# 📌 Model Mimarisi
model = Sequential([
    Input(shape=(40, 1)),
    Conv1D(64, 5, activation='relu', input_shape=(40, 1), padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(len(categories), activation='softmax')
])

# 📌 Modeli Derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 📌 Modeli Eğitme
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# 📌 Modeli Test Etme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test doğruluğu: {test_accuracy:.4f}")

# 📌 Modeli Kaydetme
model.save('voicewatch_model.h5')
print("\n✅ Model kaydedildi: voicewatch_model.h5")

# 📌 Kayıtlı Modeli Yükleme
model = load_model('voicewatch_model.h5')



# 📌 Bir Ses Dosyasını Test Etme
def predict_audio(file_path, model, label_encoder):
    feature = extract_mfcc(file_path)
    if feature is None:
        return None
    
    feature = feature.reshape(1, 40, 1)
    prediction = model.predict(feature, verbose=0)

    # 🔍 Modelin tam çıkışını terminalde yazdır
    print(f"\n🔮 Olasılık Dağılımı: {prediction[0]}")  

    class_probabilities = prediction[0]
    sorted_indices = np.argsort(class_probabilities)[::-1]
    
    top_idx = sorted_indices[0]
    confidence = class_probabilities[top_idx]
    
    if confidence > 0.5:
        class_name = label_encoder.inverse_transform([top_idx])[0]
        return class_name, confidence
    return None



# 📌 "mixed" Klasöründeki Sesleri Test Etme
mixed_folder = "mixed"

if os.path.exists(mixed_folder):
    test_files = [f for f in os.listdir(mixed_folder) if f.endswith('.wav')]
    
    if test_files:
        print(f"\n🔍 {len(test_files)} dosya bulundu. Tahminler yapılıyor...\n")
        for filename in sorted(test_files):
            file_path = os.path.join(mixed_folder, filename)
            prediction = predict_audio(file_path, model, label_encoder)
            
            if prediction:
                class_name, confidence = prediction
                print(f"📂 Dosya: {filename} -> 🏷 Tahmin: {class_name} (Güven: {confidence:.2f})")
            else:
                print(f"⚠ Dosya: {filename} -> Tanımlanamadı!")