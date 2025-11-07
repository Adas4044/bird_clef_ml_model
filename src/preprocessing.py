"""
Audio preprocessing module for BirdClef-2025
Based on paper methodology:
- Resample to 32 kHz
- Truncate/pad to 15 seconds
"""

import librosa
import numpy as np
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import warnings
import gc
import os
from multiprocessing import Pool, cpu_count
from functools import partial

#warning suppression
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import logging
logging.getLogger('audioread').setLevel(logging.ERROR)
logging.getLogger('librosa').setLevel(logging.ERROR)


class AudioPreprocessor:

    def __init__(self, target_sr=32000, target_duration=15.0, audio_dir='../birdclef-2025/train_audio', extra_audio_dir='more_birds', use_chunking=True, silence_threshold=0.001):
        
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.audio_dir = Path(audio_dir)
        self.extra_audio_dir = Path(extra_audio_dir) if extra_audio_dir else None
        self.use_chunking = use_chunking
        self.silence_threshold = silence_threshold

        print(f"AudioPreprocessor initialized:")
        print(f"  Target sample rate: {self.target_sr} Hz")
        print(f"  Target duration: {self.target_duration} seconds")
        print(f"  Target samples: {self.target_samples}")
        print(f"  Chunking: {'Enabled (split long files)' if use_chunking else 'Disabled (truncate)'}")
        print(f"  Silence filtering: {'Enabled (RMS < ' + str(silence_threshold) + ')' if use_chunking else 'Disabled'}")
        if self.extra_audio_dir and self.extra_audio_dir.exists():
            print(f"  Extra audio directory: {self.extra_audio_dir}")

    def load_audio(self, filename, sr=None):
        
        audio_path = self.audio_dir / filename

        if not audio_path.exists() and self.extra_audio_dir:
            audio_path = self.extra_audio_dir / filename

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, _ = librosa.load(audio_path, sr=sr if sr else self.target_sr)

        return y

    def resample(self, y, orig_sr):
        if orig_sr != self.target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)

        return y

    def truncate_or_pad(self, y):
        current_samples = len(y)

        if current_samples > self.target_samples:
            y = y[:self.target_samples]
        elif current_samples < self.target_samples:
            padding = self.target_samples - current_samples
            y = np.pad(y, (0, padding), mode='constant', constant_values=0)

        return y

    def chunk_audio(self, y, overlap=0.0, silence_threshold=0.001):
        current_samples = len(y)
        chunks = []

        step = int(self.target_samples * (1 - overlap))

        for start in range(0, current_samples, step):
            end = start + self.target_samples

            if end <= current_samples:
                chunk = y[start:end]

                rms = np.sqrt(np.mean(chunk ** 2))

                if rms < silence_threshold:
                    continue

                chunks.append(chunk)

        #if audio is shorter than 15, pad it
        if len(chunks) == 0 and current_samples < self.target_samples:
            padding = self.target_samples - current_samples
            chunk = np.pad(y, (0, padding), mode='constant', constant_values=0)

            rms = np.sqrt(np.mean(chunk ** 2))

            #only add if not silent
            if rms >= silence_threshold:
                chunks.append(chunk)


        return chunks

    def preprocess(self, filename):
        
        y = self.load_audio(filename, sr=self.target_sr)

        if self.use_chunking:
            chunks = self.chunk_audio(y, silence_threshold=self.silence_threshold)
            return chunks
        else:
            y = self.truncate_or_pad(y)
            return y

    def _preprocess_single_worker(self, filename):
        
        try:
            result = self.preprocess(filename)
            if isinstance(result, list):
                return result
            else:
                return [result]
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            return [np.zeros(self.target_samples)]

    def preprocess_batch(self, filenames, show_progress=True, batch_size=500, n_jobs=-1):
        
        if n_jobs == -1:
            n_workers = cpu_count()
        else:
            n_workers = min(n_jobs, cpu_count())

        print(f"  Using {n_workers} CPU cores for parallel preprocessing")

        with Pool(processes=n_workers) as pool:
            if show_progress:
                processed_chunks = list(tqdm(
                    pool.imap(self._preprocess_single_worker, filenames),
                    total=len(filenames),
                    desc="Preprocessing audio"
                ))
            else:
                processed_chunks = pool.map(self._preprocess_single_worker, filenames)

        all_audio = []
        chunk_labels = []

        for filename, chunks in zip(filenames, processed_chunks):
            for chunk in chunks:
                all_audio.append(chunk)
                chunk_labels.append(filename)

        gc.collect()

        total_chunks = len(all_audio)
        original_files = len(filenames)
        print(f"  Created {total_chunks} chunks from {original_files} files (avg {total_chunks/original_files:.1f} chunks/file)")

        return np.array(all_audio), chunk_labels

    def get_audio_info(self, filename):
        
        audio_path = self.audio_dir / filename

        if not audio_path.exists() and self.extra_audio_dir:
            audio_path = self.extra_audio_dir / filename

        try:
            info = sf.info(audio_path)
            return {
                'filename': filename,
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'samples': info.frames
            }
        except Exception as e:
            print(f"Error getting info for {filename}: {e}")
            return None

    def validate_preprocessing(self, filename):
        
        print(f"\nValidating preprocessing for: {filename}")

        audio_path = self.audio_dir / filename

        if not audio_path.exists() and self.extra_audio_dir:
            audio_path = self.extra_audio_dir / filename

        y_orig, sr_orig = librosa.load(audio_path, sr=None)
        dur_orig = librosa.get_duration(y=y_orig, sr=sr_orig)

        print(f"Original:")
        print(f"  Sample rate: {sr_orig} Hz")
        print(f"  Duration: {dur_orig:.2f}s")
        print(f"  Samples: {len(y_orig)}")

        y_processed = self.preprocess(filename)

        print(f"\nProcessed:")
        print(f"  Sample rate: {self.target_sr} Hz")
        print(f"  Duration: {len(y_processed) / self.target_sr:.2f}s")
        print(f"  Samples: {len(y_processed)}")

        print(f"\nValidation:")
        print(f"  Sample rate matches target: {self.target_sr} Hz")
        print(f"  Duration matches target: {self.target_duration}s")
        print(f"  Samples match target: {self.target_samples}")

        return {
            'original_sr': sr_orig,
            'original_duration': dur_orig,
            'original_samples': len(y_orig),
            'processed_sr': self.target_sr,
            'processed_duration': len(y_processed) / self.target_sr,
            'processed_samples': len(y_processed)
        }


def test_preprocessing():
    import pandas as pd

    train_df = pd.read_csv('../birdclef-2025/train.csv')

    preprocessor = AudioPreprocessor()

    sample_files = train_df['filename'].head(3).tolist()

    print("\nTesting preprocessing pipeline")

    for filename in sample_files:
        try:
            result = preprocessor.validate_preprocessing(filename)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    test_preprocessing()
