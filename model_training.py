# model_training.py

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling1D
from data_preprocessing import process_dataset, extract_mfcc, analyze_dataset

# Önce dataset'i analiz et
print("Dataset analiz ediliyor...")
dataset_stats = analyze_dataset('dataset2')

# Veriyi yükle ve işle
print("\nDataset işleniyor...")
all_features, all_labels = process_dataset('dataset2')

if not all_features:
    print("Hata: Dataset işlenemedi!")
    exit()

print("\nModel eğitimi başlıyor...")

# Veriyi hazırla
X = np.array(all_features)
y = np.array(all_labels)

# Etiketleri encode et
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# Veriyi yeniden şekillendir
X = X.reshape(-1, 40, 1)

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print("\nVeri boyutları:")
print(f"Eğitim seti: {X_train.shape}")
print(f"Test seti: {X_test.shape}")

# Model mimarisini güncelle
model = Sequential([
    # Giriş katmanı
    Conv1D(64, 5, activation='relu', input_shape=(40, 1), padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    
    # İlk konvolüsyon bloğu
    Conv1D(128, 5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # İkinci konvolüsyon bloğu
    Conv1D(256, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Üçüncü konvolüsyon bloğu
    Conv1D(512, 3, activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dropout(0.4),
    
    # Yoğun katmanlar
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(y_onehot.shape[1], activation='softmax')
])

# Eğitim parametrelerini güncelle
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modeli eğit
print("\nModel eğitiliyor...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
    ]
)

# Test et
print("\nModel test ediliyor...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {test_accuracy:.4f}")

# Modeli kaydet
model.save('voicewatch_model.h5')
print("\nModel kaydedildi: voicewatch_model.h5")

def test_audio_file(file_path, model, label_encoder):
    """Ses dosyasını sınıflandır"""
    feature = extract_mfcc(file_path)
    if feature is None:
        return None
    
    feature = feature.reshape(1, 40, 1)
    prediction = model.predict(feature, verbose=0)
    
    # Tüm tahminleri sırala ve güven skorlarını hesapla
    class_probabilities = prediction[0]
    sorted_indices = np.argsort(class_probabilities)[::-1]
    
    # En yüksek 3 tahmini göster
    top_predictions = []
    for idx in sorted_indices[:3]:
        class_name = label_encoder.inverse_transform([idx])[0]
        confidence = class_probabilities[idx]
        if confidence > 0.1:  # Sadece %10'dan yüksek güven skorlarını göster
            top_predictions.append((class_name, confidence))
    
    return top_predictions

# Mixed klasörünü test et
print("\nMixed klasörü test ediliyor...")
mixed_folder = "mixed"

if os.path.exists(mixed_folder):
    test_files = [f for f in os.listdir(mixed_folder) if f.endswith('.wav')]
    if test_files:
        print(f"\nToplam {len(test_files)} dosya bulundu.")
        for filename in sorted(test_files):
            file_path = os.path.join(mixed_folder, filename)
            print(f"\nDosya: {filename}")
            
            predictions = test_audio_file(file_path, model, label_encoder)
            
            if predictions:
                print("Tahminler:")
                for i, (class_name, confidence) in enumerate(predictions, 1):
                    print(f"{i}. {class_name:<15} (Güven: {confidence:.2f})")
            else:
                print("Dosya analiz edilemedi!")
