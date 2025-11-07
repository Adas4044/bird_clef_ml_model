import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import json


def load_data(data_dir='../birdclef-2025', extra_data_dir='more_birds'):

    data_path = Path(data_dir)

    train_df = pd.read_csv(data_path / 'train.csv')
    taxonomy_df = pd.read_csv(data_path / 'taxonomy.csv')

    print(f"Loaded {len(train_df)} training samples from main dataset")

    extra_path = Path(extra_data_dir)
    if extra_path.exists():
        print(f"\nLoading additional data from {extra_data_dir}...")

        extra_taxonomy_path = extra_path / 'birds_only.csv'
        if extra_taxonomy_path.exists():
            extra_taxonomy = pd.read_csv(extra_taxonomy_path)

            extra_samples = []
            for species_dir in sorted(extra_path.iterdir()):
                if species_dir.is_dir():
                    species_label = species_dir.name

                    audio_files = list(species_dir.glob('*.mp3')) + \
                                 list(species_dir.glob('*.ogg')) + \
                                 list(species_dir.glob('*.wav'))

                    for audio_file in audio_files:
                        relative_path = f"{species_label}/{audio_file.name}"
                        extra_samples.append({
                            'filename': relative_path,
                            'primary_label': species_label,
                            'source': 'more_birds'
                        })

            if extra_samples:
                extra_df = pd.DataFrame(extra_samples)
                print(f"Found {len(extra_df)} additional samples from {extra_df['primary_label'].nunique()} species")

                train_df['source'] = 'main'

                train_df = pd.concat([train_df, extra_df], ignore_index=True)

                for _, species_info in extra_taxonomy.iterrows():
                    if species_info['primary_label'] not in taxonomy_df['primary_label'].values:
                        taxonomy_df = pd.concat([taxonomy_df, pd.DataFrame([species_info])], ignore_index=True)

                print(f"Combined total: {len(train_df)} samples from {train_df['primary_label'].nunique()} species")

    print(f"Final dataset: {len(taxonomy_df)} species in taxonomy")

    return train_df, taxonomy_df


def stratified_sample(df, label_col='primary_label', min_samples=500,
                     max_samples=500, random_state=42, include_all_species=False):
    
    balanced_dfs = []
    skipped_species = []

    for label in df[label_col].unique():
        label_df = df[df[label_col] == label]
        n_samples = len(label_df)

        if n_samples < min_samples and not include_all_species:
            print(f"Skipping: {label} has only {n_samples} samples (< {min_samples})")
            skipped_species.append(label)
            continue
        elif n_samples < min_samples and include_all_species:
            print(f"Including: {label} with {n_samples} samples (< {min_samples})")
            balanced_dfs.append(label_df)
        elif n_samples > max_samples:
            sampled = label_df.sample(n=max_samples, random_state=random_state)
            balanced_dfs.append(sampled)
        else:
            balanced_dfs.append(label_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"\nBalanced dataset: {len(balanced_df)} samples")
    print(f"Species included: {balanced_df[label_col].nunique()}")
    if skipped_species:
        print(f"Species skipped: {len(skipped_species)}")
    print(f"Species distribution: {balanced_df[label_col].value_counts().describe()}")

    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def evaluate_model(y_true, y_pred, model_name='Model'):
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    print(f"{model_name} Performance")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    return metrics


def plot_confusion_matrix(y_true, y_pred, labels=None, model_name='Model',
                         save_path=None, figsize=(12, 10)):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)

    if labels is not None and len(labels) < 50:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    else:
        sns.heatmap(cm, annot=False, cmap='Blues')

    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def compare_models(metrics_list, save_path=None):
    df = pd.DataFrame(metrics_list)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.2

    metrics_to_plot = ['precision', 'recall', 'f1']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        offset = width * (i - 1)
        ax.bar(x + offset, df[metric], width, label=metric.capitalize(), color=color)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        offset = width * (i - 1)
        for j, v in enumerate(df[metric]):
            ax.text(j + offset, v + 0.01, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    return df


def save_metrics(metrics, filepath):
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath):
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def print_dataset_info(df, taxonomy_df):
    print("dataset info")

    print(f"\nTotal samples: {len(df)}")
    print(f"Total species: {df['primary_label'].nunique()}")

    merged = df.merge(taxonomy_df[['primary_label', 'class_name']],
                     on='primary_label', how='left')

    print(f"\nClass distribution:")
    for class_name, count in merged['class_name'].value_counts().items():
        print(f"  {class_name}: {count} samples")

    print(f"\nSamples per species:")
    species_counts = df['primary_label'].value_counts()
    print(f"Min: {species_counts.min()}")
    print(f"Max: {species_counts.max()}")
    print(f"Mean: {species_counts.mean():.2f}")
    print(f"Median: {species_counts.median():.2f}")