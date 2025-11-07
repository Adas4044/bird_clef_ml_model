import librosa
import numpy as np
from tqdm import tqdm
import warnings
import gc
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


class MFCCExtractor:
    
    def __init__(self, sr=32000, n_mfcc=40, n_fft=2048, hop_length=512):
        
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

        print(f"MFCCExtractor initialized:")
        print(f"Sample rate: {self.sr} Hz")
        print(f"Number of MFCCs: {self.n_mfcc}")
        print(f"FFT size: {self.n_fft}")
        print(f"Hop length: {self.hop_length}")

    def extract_mfcc(self, y):
       
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        return mfccs

    def extract_mfcc_statistics(self, y):
        
        mfccs = self.extract_mfcc(y)

        mfcc_mean = np.mean(mfccs, axis=1)

        return mfcc_mean

    def extract_mfcc_extended_statistics(self, y):
        
        mfccs = self.extract_mfcc(y)

        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_min = np.min(mfccs, axis=1)
        mfcc_max = np.max(mfccs, axis=1)

        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])

        return features

    def _extract_single_worker(self, args):
        
        y, statistics = args
        try:
            if statistics == 'mean':
                return self.extract_mfcc_statistics(y)
            elif statistics == 'extended':
                return self.extract_mfcc_extended_statistics(y)
            else:
                raise ValueError(f"Unknown statistics type: {statistics}")
        except Exception as e:
            print(f"\nError extracting features: {e}")
            n_features = self.n_mfcc if statistics == 'mean' else self.n_mfcc * 4
            return np.zeros(n_features)

    def extract_batch(self, audio_batch, statistics='mean', show_progress=True, batch_size=500, n_jobs=-1):
        
        if n_jobs == -1:
            n_workers = cpu_count()
        else:
            n_workers = min(n_jobs, cpu_count())

        print(f"  Using {n_workers} CPU cores for parallel feature extraction")

        args_list = [(y, statistics) for y in audio_batch]

        with Pool(processes=n_workers) as pool:
            if show_progress:
                features = list(tqdm(
                    pool.imap(self._extract_single_worker, args_list),
                    total=len(args_list),
                    desc="Extracting features"
                ))
            else:
                features = pool.map(self._extract_single_worker, args_list)

        gc.collect()

        return np.array(features)

    def visualize_mfcc(self, y, title='MFCC Features'):
        
        import matplotlib.pyplot as plt

        mfccs = self.extract_mfcc(y)

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mfccs, sr=self.sr, x_axis='time', hop_length=self.hop_length)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.ylabel('MFCC Coefficients')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def compare_statistics(self, y):
        
        mfccs = self.extract_mfcc(y)

        stats = {
            'mean': np.mean(mfccs, axis=1),
            'std': np.std(mfccs, axis=1),
            'min': np.min(mfccs, axis=1),
            'max': np.max(mfccs, axis=1),
            'median': np.median(mfccs, axis=1),
            'q25': np.percentile(mfccs, 25, axis=1),
            'q75': np.percentile(mfccs, 75, axis=1)
        }

        return stats


def extract_additional_features(y, sr=32000):
    
    features = {}

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)

    return features


def test_feature_extraction():
    """Test feature extraction on sample audio"""
    from preprocessing import AudioPreprocessor
    import pandas as pd

    print("\nTesting MFCC extraction pipeline...")

    train_df = pd.read_csv('../birdclef-2025/train.csv')

    preprocessor = AudioPreprocessor()
    extractor = MFCCExtractor()

    filename = train_df['filename'].iloc[0]
    print(f"\nProcessing: {filename}")

    y = preprocessor.preprocess(filename)
    print(f"Audio shape: {y.shape}")

    features_mean = extractor.extract_mfcc_statistics(y)
    print(f"\nMFCC features (mean): {features_mean.shape}")
    print(f"Feature values (first 5): {features_mean[:5]}")

    features_extended = extractor.extract_mfcc_extended_statistics(y)
    print(f"\nMFCC features (extended): {features_extended.shape}")
    print(f"Feature breakdown: {extractor.n_mfcc} MFCCs x 4 statistics = {len(features_extended)}")

    print("\nVisualizing MFCC features...")
    extractor.visualize_mfcc(y, title=f'MFCC Features - {filename}')

    print("\nTesting batch extraction...")
    sample_files = train_df['filename'].head(5).tolist()
    audio_batch = preprocessor.preprocess_batch(sample_files, show_progress=True)
    features_batch = extractor.extract_batch(audio_batch, statistics='mean', show_progress=True)

    print(f"\nBatch features shape: {features_batch.shape}")
    print(f"  {features_batch.shape[0]} samples")
    print(f"  {features_batch.shape[1]} features per sample")


if __name__ == "__main__":
    test_feature_extraction()
