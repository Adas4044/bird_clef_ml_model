import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from datetime import datetime
import sys

sys.path.append('src')

from preprocessing import AudioPreprocessor
from feature_extraction import MFCCExtractor
from models import BirdClassifier


st.set_page_config(
    page_title="BirdClef-2025 Species Classifier",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #fff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load pre-trained models"""
    models_dir = Path('outputs/models')

    if not models_dir.exists():
        return None

    models = {}

    # Try to load each model
    for model_file in models_dir.glob('*.pkl'):
        model_name = model_file.stem.replace('_model', '').upper()
        try:
            classifier = BirdClassifier(model_type='knn')  # Dummy init
            classifier.load(model_file)
            models[model_name] = classifier
        except Exception as e:
            st.warning(f"Could not load {model_name}: {e}")

    return models if models else None


@st.cache_resource
def load_taxonomy():
    """Load taxonomy data"""
    try:
        taxonomy_df = pd.read_csv('birdclef-2025/taxonomy.csv')
        return taxonomy_df
    except:
        return None


def process_audio(audio_file):
    """
    Process uploaded audio file

    Args:
        audio_file: Uploaded file object

    Returns:
        tuple: (preprocessed audio, sample rate)
    """
    # Save temporary file
    temp_path = Path('temp_audio.ogg')
    with open(temp_path, 'wb') as f:
        f.write(audio_file.getbuffer())

    # Load and preprocess
    preprocessor = AudioPreprocessor()
    y, sr = librosa.load(temp_path, sr=preprocessor.target_sr)
    y = preprocessor.truncate_or_pad(y)

    # Clean up
    temp_path.unlink()

    return y, sr


def extract_features(y):
    """Extract MFCC features from audio"""
    extractor = MFCCExtractor()
    features = extractor.extract_mfcc_statistics(y)
    return features.reshape(1, -1)


def plot_audio_waveform(y, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    return fig


def plot_mel_spectrogram(y, sr):
    """Plot mel-spectrogram"""
    fig, ax = plt.subplots(figsize=(10, 4))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-frequency Spectrogram')
    return fig


def main():
    """Main Streamlit app"""

    # Header
    st.markdown('<h1 class="main-header">🦜 Bird Species Prediction</h1>', unsafe_allow_html=True)
    st.markdown("Upload an audio file to predict the bird species using Machine Learning models")

    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        This application uses Machine Learning to classify bird species from audio recordings.

        **Models:**
        - K-Nearest Neighbors (KNN)
        - Random Forest
        - Support Vector Machine (SVM)

        **Features:**
        - MFCC-based audio analysis
        - Real-time predictions
        - Audio visualization
        """)

        st.header("⚙️ Settings")
        selected_model = st.selectbox(
            "Select Model",
            ["Random Forest", "KNN", "SVM"],
            index=0
        )

        show_visualization = st.checkbox("Show Visualizations", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)

    # Load models and taxonomy
    models = load_models()
    taxonomy_df = load_taxonomy()

    if models is None:
        st.error("⚠️ No trained models found. Please train models first using `python src/train.py`")
        st.stop()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📁 Upload Audio")

        uploaded_file = st.file_uploader(
            "Choose an audio file (.ogg, .wav, .mp3)",
            type=['ogg', 'wav', 'mp3'],
            help="Upload a bird vocalization recording"
        )

        if uploaded_file is not None:
            st.success(f"✓ File uploaded: {uploaded_file.name}")

            # Process audio
            with st.spinner("Processing audio..."):
                try:
                    y, sr = process_audio(uploaded_file)
                    st.audio(uploaded_file, format='audio/ogg')

                    # Display audio info
                    st.info(f"""
                    **Audio Information:**
                    - Sample Rate: {sr} Hz
                    - Duration: {len(y)/sr:.2f} seconds
                    - Samples: {len(y)}
                    """)

                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    st.stop()

            # Extract features
            with st.spinner("Extracting features..."):
                try:
                    features = extract_features(y)
                    st.success("✓ Features extracted successfully")
                except Exception as e:
                    st.error(f"Error extracting features: {e}")
                    st.stop()

            # Make prediction
            with st.spinner(f"Making prediction using {selected_model}..."):
                try:
                    model_key = selected_model.upper().replace(' ', '_')
                    if model_key == 'RANDOM_FOREST':
                        model_key = 'RANDOM'  # Match saved model name

                    if model_key in models:
                        classifier = models[model_key]
                        prediction = classifier.predict(features)[0]

                        # Get taxonomy info
                        if taxonomy_df is not None:
                            species_info = taxonomy_df[taxonomy_df['primary_label'] == prediction]

                            if not species_info.empty:
                                common_name = species_info.iloc[0]['common_name']
                                scientific_name = species_info.iloc[0]['scientific_name']
                                class_name = species_info.iloc[0]['class_name']
                            else:
                                common_name = prediction
                                scientific_name = "Unknown"
                                class_name = "Unknown"
                        else:
                            common_name = prediction
                            scientific_name = "Unknown"
                            class_name = "Unknown"

                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.header("🎯 Prediction Results")
                        st.markdown(f"### **{common_name}**")
                        st.markdown(f"*{scientific_name}*")
                        st.markdown(f"**Class:** {class_name}")
                        st.markdown(f"**Model:** {selected_model}")
                        st.markdown(f"**Predicted at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Show confidence scores if supported
                        if show_confidence and hasattr(classifier.model, 'predict_proba'):
                            try:
                                proba = classifier.predict_proba(features)[0]
                                top_indices = np.argsort(proba)[-5:][::-1]

                                st.subheader("📊 Top 5 Predictions")

                                for idx in top_indices:
                                    species = classifier.label_encoder.inverse_transform([idx])[0]
                                    confidence = proba[idx] * 100

                                    if taxonomy_df is not None:
                                        species_info = taxonomy_df[taxonomy_df['primary_label'] == species]
                                        if not species_info.empty:
                                            display_name = species_info.iloc[0]['common_name']
                                        else:
                                            display_name = species
                                    else:
                                        display_name = species

                                    st.progress(confidence / 100)
                                    st.text(f"{display_name}: {confidence:.2f}%")

                            except Exception as e:
                                st.warning(f"Could not compute confidence scores: {e}")

                    else:
                        st.error(f"Model {selected_model} not found in loaded models")

                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.stop()

            # Visualizations
            if show_visualization:
                st.header("📈 Audio Visualizations")

                tab1, tab2 = st.tabs(["Waveform", "Mel-Spectrogram"])

                with tab1:
                    fig_wave = plot_audio_waveform(y, sr)
                    st.pyplot(fig_wave)

                with tab2:
                    fig_mel = plot_mel_spectrogram(y, sr)
                    st.pyplot(fig_mel)

    with col2:
        st.header("📖 Instructions")
        st.markdown("""
        1. **Select a model** from the sidebar
        2. **Upload an audio file** containing bird vocalizations
        3. **View the prediction** and confidence scores
        4. **Explore visualizations** to understand the audio

        **Supported formats:**
        - OGG
        - WAV
        - MP3

        **Tips:**
        - Use clear recordings with minimal background noise
        - Audio will be automatically resampled to 32 kHz
        - Recordings will be truncated/padded to 15 seconds
        """)

        # Model performance
        if Path('outputs/model_comparison.csv').exists():
            st.header("🏆 Model Performance")
            comparison_df = pd.read_csv('outputs/model_comparison.csv')
            st.dataframe(comparison_df, hide_index=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🤖 Generated with <a href='https://claude.com/claude-code'>Claude Code</a> |
        Based on BirdClef-2025 Dataset |
        Reimplementation of "Bird Species Classification Using Machine Learning"
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
