# model_training.py

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

# data_preprocessing.py dosyanızdan all_features ve all_labels içe aktarılıyor.
from data_preprocessing import all_features, all_labels

# 1. Listeleri NumPy dizilerine dönüştürme
X = np.array(all_features)  # (num_samples, 40)
y = np.array(all_labels)    # (num_samples,)

# 2. Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. One-hot encoding
y_onehot = to_categorical(y_encoded)

# 4. Giriş verisini yeniden şekillendirme: (num_samples, 40, 1)
X = X.reshape(-1, 40, 1)

# 5. Eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# 6. Model Tasarımı
input_shape = (40, 1)
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Model Eğitimi
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)

# 8. Test Setinde Değerlendirme
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 9. Modeli Kaydetme
model.save('voicewatch_model.h5')
print("Model kaydedildi: voicewatch_model.h5")
