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
CATEGORIES = ['glass_breaking', 'fall', 'silence', 'scream']
MIXED_PATH = 'mixed'

SR = 16000            # Sampling rate (Hz)
DURATION = 2          # Duration of each audio clip in seconds
SAMPLES_PER_FILE = SR * DURATION
N_MELS = 128          # Number of Mel bands

# Function to extract mel-spectrogram (no saving to file)
def extract_mel(file_path):
    audio, _ = librosa.load(file_path, sr=SR)
    if len(audio) < SAMPLES_PER_FILE:
        audio = np.pad(audio, (0, SAMPLES_PER_FILE - len(audio)))  # Zero-padding if too short
    else:
        audio = audio[:SAMPLES_PER_FILE]  # Truncate if too long
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)  # Convert to dB scale
    return mel_db

# Load dataset and labels from folders
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

# Build a simple CNN model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((N_MELS, -1, 1)),  # Add channel dimension
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

# Main pipeline
def main():
    print("Loading dataset...")
    X, y = load_dataset()

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create CNN model
    model = create_model(X_train.shape[1:], len(CATEGORIES))
    model.summary()

    # Train the model
    print("Training model...")
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_val, y_val))

    # Select and predict 3 random files from 'mixed' folder
    print("\nPredicting 3 random audio files from 'mixed' folder...")
    all_files = [f for f in os.listdir(MIXED_PATH) if f.endswith('.wav')]
    selected_files = random.sample(all_files, min(3, len(all_files)))

    for file_name in selected_files:
        file_path = os.path.join(MIXED_PATH, file_name)
        mel_db = extract_mel(file_path)
        mel_db_batch = np.expand_dims(mel_db, axis=0)  # Add batch dimension
        prediction = model.predict(mel_db_batch)
        predicted_class = np.argmax(prediction)
        print(f"{file_name} -> Prediction: {CATEGORIES[predicted_class]}")

        # Plot Mel-spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=SR, hop_length=512, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram - {file_name}\nPredicted: {CATEGORIES[predicted_class]}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
