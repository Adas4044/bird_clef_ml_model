import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import json
from datetime import datetime
from multiprocessing import cpu_count

from preprocessing import AudioPreprocessor
from feature_extraction import MFCCExtractor
from models import BirdClassifier, train_all_models
from utils import (load_data, stratified_sample, evaluate_model,
                  compare_models, save_metrics, print_dataset_info)


def main(args):
    print("BirdClef-2025 Classifier Training Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU Cores Available: {cpu_count()}")
    print(f"Parallel Processing: {'Enabled' if args.n_jobs != 1 else 'Disabled'}")
    if args.n_jobs == -1:
        print(f"Using all {cpu_count()} CPU cores")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Step 1: Loading Data")

    extra_data_path = Path(args.data_dir).parent / 'more_birds'

    train_df, taxonomy_df = load_data(args.data_dir, extra_data_dir=str(extra_data_path))
    print_dataset_info(train_df, taxonomy_df)

    print("Step 2: Stratified Sampling")

    if args.use_sampling:
        if args.include_all_species:
            print(f"stratified sampling - ALL SPECIES (max={args.max_samples} per species)")
        else:
            print(f"stratified sampling (min={args.min_samples}, max={args.max_samples})")
        balanced_df = stratified_sample(
            train_df,
            label_col='primary_label',
            min_samples=args.min_samples,
            max_samples=args.max_samples,
            random_state=args.random_state,
            include_all_species=args.include_all_species
        )
    else:
        print("Skipping sampling - using full dataset")
        balanced_df = train_df

    # Limit dataset size if specified
    if args.max_total_samples and len(balanced_df) > args.max_total_samples:
        print(f"Limiting dataset to {args.max_total_samples} samples")
        balanced_df = balanced_df.sample(n=args.max_total_samples, random_state=args.random_state)

    print(f"\nFinal dataset size: {len(balanced_df)} samples")

    preprocessor = AudioPreprocessor(
        target_sr=args.target_sr,
        target_duration=args.target_duration,
        audio_dir=Path(args.data_dir) / 'train_audio',
        extra_audio_dir=str(extra_data_path),
        use_chunking=args.use_chunking,
        silence_threshold=args.silence_threshold
    )

    print("Step 3-4: Preprocessing and Feature Extraction")

    extractor = MFCCExtractor(
        sr=args.target_sr,
        n_mfcc=args.n_mfcc
    )

    filenames = balanced_df['filename'].tolist()
    labels_original = balanced_df['primary_label'].tolist()

    chunk_size = 1000
    all_features = []
    all_labels = []

    print(f"Processing {len(filenames)} files in chunks of {chunk_size}")

    for i in range(0, len(filenames), chunk_size):
        chunk_files = filenames[i:i+chunk_size]
        chunk_labels = labels_original[i:i+chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(filenames) + chunk_size - 1) // chunk_size

        print(f"\nChunk {chunk_num}/{total_chunks} ({len(chunk_files)} files)")

        audio_chunks, chunk_filenames = preprocessor.preprocess_batch(chunk_files, show_progress=True, n_jobs=args.n_jobs)

        filename_to_label = dict(zip(chunk_files, chunk_labels))
        expanded_labels = [filename_to_label[fn] for fn in chunk_filenames]

        features_chunk = extractor.extract_batch(
            audio_chunks,
            statistics=args.feature_statistics,
            show_progress=True,
            n_jobs=args.n_jobs
        )

        all_features.append(features_chunk)
        all_labels.extend(expanded_labels)

        del audio_chunks
        del features_chunk
        del expanded_labels
        import gc
        gc.collect()

    features = np.vstack(all_features)
    labels = np.array(all_labels)
    del all_features
    del all_labels
    gc.collect()

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"  Total audio chunks: {features.shape[0]}")
    print(f"  Features per chunk: {features.shape[1]}")
    print(f"  Original files: {len(filenames)}")
    print(f"  Expansion ratio: {features.shape[0] / len(filenames):.2f}x")

   
    print("Step 4: Train/Test Split")

    from collections import Counter
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    if min_count >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=labels
        )
        print(f"Using stratified split (min samples per class: {min_count})")
    else:
        print(f"Some classes have only {min_count} samples. Using random split instead.")
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=args.test_size,
            random_state=args.random_state
        )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train/Test ratio: {args.test_size*100:.0f}%")

    split_info = {
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'n_classes': int(len(np.unique(labels))),
        'test_ratio': float(args.test_size)
    }

    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=4)

    
    print("Step 5: Model Training and Evaluation")

    results = train_all_models(
        X_train, X_test, y_train, y_test,
        save_dir=output_dir / 'models'
    )

    print("Step 6: Model Comparison")

    metrics_list = [results[model]['metrics'] for model in results]

    comparison_df = compare_models(
        metrics_list,
        save_path=output_dir / 'model_comparison.png'
    )

    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\nComparison results saved to {output_dir / 'model_comparison.csv'}")

    for model_name, result in results.items():
        metrics_file = output_dir / f'{model_name.lower().replace(" ", "_")}_metrics.json'
        save_metrics(result['metrics'], metrics_file)

   
    print("Training Summary")

    print(f"\nDataset:")
    print(f"Total samples: {len(balanced_df)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Classes: {len(np.unique(labels))}")
    print(f"Features: {X_train.shape[1]}")

    print(f"\nModel Performance:")
    for model_name in results:
        metrics = results[model_name]['metrics']
        print(f"\n  {model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

    best_model = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    print(f"\n  Best Model: {best_model[0]} (Accuracy: {best_model[1]['metrics']['accuracy']:.4f})")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"Models: {output_dir / 'models'}")
    print(f"Metrics: {output_dir / '*.json'}")
    print(f"Plots: {output_dir / '*.png'}")

    print(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BirdClef-2025 classification models")

    # Data paths
    parser.add_argument('--data-dir', type=str, default='../data/birdclef-2025')
    parser.add_argument('--output-dir', type=str, default='../outputs')

    # Sampling configuration
    parser.add_argument('--use-sampling', action='store_true', default=True)
    parser.add_argument('--min-samples', type=int, default=50)
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--max-total-samples', type=int, default=None)
    parser.add_argument('--include-all-species', action='store_true', default=False)

    # Audio preprocessing
    parser.add_argument('--target-sr', type=int, default=32000)
    parser.add_argument('--target-duration', type=float, default=15.0)
    parser.add_argument('--use-chunking', action='store_true', default=True)
    parser.add_argument('--no-chunking', dest='use_chunking', action='store_false')
    parser.add_argument('--silence-threshold', type=float, default=0.001)

    # Feature extraction
    parser.add_argument('--n-mfcc', type=int, default=40)
    parser.add_argument('--feature-statistics', type=str, default='mean', choices=['mean', 'extended'])

    # Training configuration
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)

    args = parser.parse_args()
    results = main(args)
