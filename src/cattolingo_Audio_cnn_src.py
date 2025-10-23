import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import tensorflow as tf
import os
from huggingface_hub import hf_hub_download 

CLASS_NAMES = ["Angry", "Happy", "Stressed", "Sad", "Resting"]


HF_REPO_ID_AUDIO = "Nour87/cattolingo-audio-cnn" 
MODEL_FILENAME_AUDIO = "cattolingo-Audio_cnn-model.h5"

def load_cnn_model():
 
    try:
        print(f"[Audio SRC] Downloading model '{MODEL_FILENAME_AUDIO}' from repo '{HF_REPO_ID_AUDIO}'...")
       
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID_AUDIO,
            filename=MODEL_FILENAME_AUDIO
        )
        print(f"[Audio SRC] Model downloaded to: {model_path}")

        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(" CNN Cat Emotion Model loaded successfully from Hub file (safe mode).")
        return model

    except Exception as e:
        print(f" Error loading model from Hub: {e}") 
        return None


def create_spectrogram_tensor(audio_file, target_size=(232, 231)):
    try:
        y, sr = librosa.load(audio_file, sr=44100)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            hop_length=256,
            fmax=20000
        )
        S_DB = librosa.power_to_db(S, ref=np.max)

        fig = plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_DB, sr=sr, cmap="magma")
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"❌ Error creating spectrogram tensor: {e}")
        return None

def get_prediction(model, audio_path):
    tensor = create_spectrogram_tensor(audio_path)
    if tensor is None:
        return None, None 
    try: 
        preds = model.predict(tensor)
        confidence = float(np.max(preds[0])) 
        label_index = int(np.argmax(preds[0])) 
        label = CLASS_NAMES[label_index] if label_index < len(CLASS_NAMES) else "Unknown"
        print(f"🎯 Prediction: {label} ({confidence*100:.2f}%)")
        return label, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None


if __name__ == "__main__":
    print("Testing src file... Model loading will be from Hub.")