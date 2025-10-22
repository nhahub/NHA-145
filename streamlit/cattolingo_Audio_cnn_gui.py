import sys, os
import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import io


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cattolingo_Audio_cnn_src import load_cnn_model, get_prediction, create_spectrogram_tensor, CLASS_NAMES



st.set_page_config(
    page_title="Cat Emotion Detector 😺 (V3)",
    layout="wide",
    page_icon="🐾"
)

st.markdown("<h1 style='text-align:center; color:#FF6B6B;'>🐱 Cat Emotion Detector (V3)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4ECDC4;'>Detect your cat's emotion from its meow — Improved Accuracy 🎯</p>", unsafe_allow_html=True)



@st.cache_resource
def get_model():
    with st.spinner("Loading AI model..."):
        model = load_cnn_model()
        if model is not None:
            model.compile()
        return model

model = get_model()



def audio_to_spectrogram(audio_file):
    """Use the new create_spectrogram_tensor and also save image for preview."""
    tensor = create_spectrogram_tensor(audio_file)

   
    y, sr = librosa.load(audio_file, sr=44100)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=256, fmax=20000)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(4, 3))
    librosa.display.specshow(S_DB, sr=sr, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return tensor, img


def show_waveform(audio_path):
    """Display waveform of uploaded audio."""
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, color="#FF6B6B")
    ax.set_title("Audio Waveform", color="#4ECDC4")
    st.pyplot(fig)



with st.sidebar:
    st.image("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", use_column_width=True)
    st.markdown("### 📤 Upload Audio File")
    audio_file = st.file_uploader("Upload your cat's meow (wav/mp3)", type=["wav", "mp3"])
    st.info("🎧 Model: CNN trained on Cat Meow Spectrograms (V3)")



if not model:
    st.error("❌ Model not loaded properly. Please check your model path.")
elif audio_file is None:
    st.info("⬆️ Please upload a cat audio file to start analysis.")
else:
    st.audio(audio_file)
    temp_audio = "temp_audio.wav"
    with open(temp_audio, "wb") as f:
        f.write(audio_file.getbuffer())

    show_waveform(temp_audio)

    if st.button("🔮 Predict Emotion", use_container_width=True):
        with st.spinner("Analyzing your cat's sound... ⏳"):
            try:
                tensor, spec_img = audio_to_spectrogram(temp_audio)
                label, confidence = get_prediction(model, temp_audio)

                st.markdown("### 🎯 Prediction Result")
                st.success(f"**Emotion:** {label}")
                st.info(f"**Confidence:** {confidence*100:.2f}%")

                
                st.image(spec_img, caption="Generated Spectrogram (V3)", use_container_width=True)

             
                preds = model.predict(tensor)
                probs = tf.nn.softmax(preds[0]).numpy()

                fig = go.Figure([go.Bar(
                    x=probs * 100,
                    y=CLASS_NAMES,
                    orientation='h',
                    text=[f"{p*100:.2f}%" for p in probs],
                    textposition='auto',
                    marker=dict(color=probs * 100, colorscale='Viridis')
                )])
                fig.update_layout(title="Prediction Confidence", xaxis_title="Confidence (%)")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
            finally:
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit and TensorFlow — V3 Engine</p>",
    unsafe_allow_html=True
)
