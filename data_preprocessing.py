# Ses Dosyalarını Yükleme ve Format Standartlaştırması
import librosa

def load_audio(file_path, sr=16000):
    """
    Verilen dosya yolundan ses dosyasını yükler.
    Tüm dosyalar 16kHz örnekleme oranında yüklenecektir.
    """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

import numpy as np

def extract_mfcc(file_path, n_mfcc=40):
    """
    Belirtilen ses dosyasından MFCC özniteliklerini çıkarır.
    MFCC'leri zaman ekseninde ortalayarak sabit uzunlukta bir vektör elde eder.
    """
    audio, sr = load_audio(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Tüm zaman adımlarının ortalamasını alır
    return mfcc_mean

import os

def extract_features_from_folder(folder_path, n_mfcc=40):
    """
    Belirtilen klasördeki tüm .wav dosyalarını okuyup, MFCC özniteliklerini çıkarır.
    Aynı zamanda, dosyanın ait olduğu klasör ismini etiket olarak ekler.
    """
    features = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            feature_vector = extract_mfcc(file_path, n_mfcc)
            features.append(feature_vector)
            # Klasör adı, etiket olarak kullanılabilir
            labels.append(os.path.basename(folder_path))
    return features, labels

# Dataset dizininizin yolu
dataset_path = 'dataset2'
# Klasör yapınıza göre kategori isimleri
categories = [
    'scream',
    'silence',
    'fall',
    'glass_breaking',
    os.path.join('background', 'NotScreaming')
]

all_features = []
all_labels = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    features, labels = extract_features_from_folder(folder)
    all_features.extend(features)
    all_labels.extend(labels)

print("Özellik vektörlerinin sayısı:", len(all_features))
print("Etiket örneklerinden bazıları:", all_labels[:5])

def augment_audio(audio, sr):
    """
    Verilen ses verisini kullanarak farklı augmentasyon teknikleri uygular.
    Örneğin; zaman germe ve pitch shifting.
    """
    augmented = []
    # Time stretching: sesin hızını %10 yavaşlat ve %10 hızlandır
    stretched_slow = librosa.effects.time_stretch(audio, rate=0.9)
    stretched_fast = librosa.effects.time_stretch(audio, rate=1.1)
    augmented.append(stretched_slow)
    augmented.append(stretched_fast)
    
    # Pitch shifting: sesin perdesini 2 semiton artır
    pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
    augmented.append(pitched)
    
    return augmented

# Örnek: Bir ses dosyasını augmentasyon ile zenginleştirme
if __name__ == "__main__":
    example_file = os.path.join(dataset_path, 'scream', '1440.wav')  # Burada 'ornek.wav' dosyasının var olduğundan emin olun
    audio, sr = load_audio(example_file)
    
    augmented_audios = augment_audio(audio, sr)
    print("Augmentasyon ile elde edilen örnek ses sayısı:", len(augmented_audios))
