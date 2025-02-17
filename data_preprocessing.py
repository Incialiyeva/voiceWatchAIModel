# Ses Dosyalarını Yükleme ve Format Standartlaştırması
import librosa
import numpy as np
import os

def load_audio(file_path, sr=16000):
    """
    Verilen dosya yolundan ses dosyasını yükler.
    Tüm dosyalar 16kHz örnekleme oranında yüklenecektir.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        return audio, sample_rate
    except Exception as e:
        print(f"Hata - Dosya yüklenemedi: {file_path}")
        print(f"Hata mesajı: {str(e)}")
        return None, None

def extract_mfcc(file_path, n_mfcc=40):
    """
    Belirtilen ses dosyasından MFCC özniteliklerini çıkarır.
    """
    audio, sr = load_audio(file_path)
    if audio is None:
        return None
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"MFCC çıkarma hatası - Dosya: {file_path}")
        print(f"Hata mesajı: {str(e)}")
        return None

def extract_features_from_folder(folder_path, n_mfcc=40):
    """
    Belirtilen klasördeki tüm .wav dosyalarını okuyup, MFCC özniteliklerini çıkarır.
    """
    features = []
    labels = []
    errors = []
    
    print(f"İşleniyor: {folder_path}")
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            feature_vector = extract_mfcc(file_path, n_mfcc)
            
            if feature_vector is not None:
                features.append(feature_vector)
                labels.append(os.path.basename(folder_path))
            else:
                errors.append(file_path)
    
    print(f"Klasör tamamlandı: {folder_path}")
    print(f"İşlenen dosya sayısı: {len(features)}")
    if errors:
        print(f"Hatalı dosya sayısı: {len(errors)}")
    
    return features, labels

def process_dataset(dataset_path):
    """
    Tüm dataset'i işler
    """
    all_features = []
    all_labels = []
    
    categories = [
        'scream',
        'silence',
        'fall',
        'glass_breaking',
        os.path.join('background', 'NotScreaming')
    ]
    
    print("Dataset işleniyor...")
    
    for category in categories:
        folder = os.path.join(dataset_path, category)
        if os.path.exists(folder):
            features, labels = extract_features_from_folder(folder)
            all_features.extend(features)
            all_labels.extend(labels)
        else:
            print(f"Uyarı: Klasör bulunamadı - {folder}")
    
    print(f"\nToplam işlenen örnek sayısı: {len(all_features)}")
    print(f"Etiket dağılımı:")
    unique_labels = set(all_labels)
    for label in unique_labels:
        count = all_labels.count(label)
        print(f"{label}: {count} örnek")
    
    return all_features, all_labels

def analyze_dataset(dataset_path):
    """
    Dataset'i detaylı analiz eder ve her kategorideki ses dosyalarının özelliklerini raporlar
    """
    categories = [
        'scream',
        'silence',
        'fall',
        'glass_breaking',
        os.path.join('background', 'NotScreaming')
    ]
    
    print("\nDataset Analizi:")
    print("=" * 50)
    
    total_files = 0
    category_stats = {}
    
    for category in categories:
        folder = os.path.join(dataset_path, category)
        if not os.path.exists(folder):
            print(f"Uyarı: {category} klasörü bulunamadı!")
            continue
            
        wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        num_files = len(wav_files)
        total_files += num_files
        
        print(f"\nKategori: {category}")
        print(f"Dosya sayısı: {num_files}")
        
        # Her kategoriden örnek ses dosyalarını analiz et
        sample_files = wav_files[:5]  # İlk 5 dosyayı incele
        features = []
        durations = []
        
        for wav_file in sample_files:
            file_path = os.path.join(folder, wav_file)
            try:
                audio, sr = load_audio(file_path)
                if audio is not None:
                    duration = len(audio) / sr
                    durations.append(duration)
                    
                    feature = extract_mfcc(file_path)
                    if feature is not None:
                        features.append(feature)
            except Exception as e:
                print(f"Hata - {wav_file}: {str(e)}")
                
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"Ortalama süre: {avg_duration:.2f} saniye")
        
        category_stats[category] = {
            'num_files': num_files,
            'avg_duration': avg_duration if durations else 0
        }
    
    print("\nÖzet:")
    print(f"Toplam dosya sayısı: {total_files}")
    for category, stats in category_stats.items():
        print(f"{category}: {stats['num_files']} dosya, Ort. süre: {stats['avg_duration']:.2f}s")
    
    return category_stats

if __name__ == "__main__":
    dataset_path = 'dataset2'
    stats = analyze_dataset(dataset_path)
    all_features, all_labels = process_dataset(dataset_path)

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
