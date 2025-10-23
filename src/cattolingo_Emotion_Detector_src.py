import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import cv2
from huggingface_hub import hf_hub_download
import streamlit as st
from collections import Counter
import tempfile

class VideoConfig:
    HF_REPO_ID = "Nour87/cattolingo-video"
    MODEL_FILENAME = "cattolingo-Emotion_Detector-model_FIXED.pth"
    INPUT_IMAGE_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEFAULT_CLASSES = {
        0: 'angry', 1: 'disgusted', 2: 'happy', 3: 'normal',
        4: 'relaxed', 5: 'sad', 6: 'scared', 7: 'surprised', 8: 'uncomfortable'
    }
    NUM_FIXED_CLASSES = len(DEFAULT_CLASSES)

@st.cache_resource
def load_classification_model():
    try:
        print(f"Downloading model '{VideoConfig.MODEL_FILENAME}' from repo '{VideoConfig.HF_REPO_ID}'...")
        model_weights_path = hf_hub_download(
            repo_id=VideoConfig.HF_REPO_ID,
            filename=VideoConfig.MODEL_FILENAME
        )
        print(f"Model downloaded to: {model_weights_path}")

        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, VideoConfig.NUM_FIXED_CLASSES)

        state_dict = torch.load(model_weights_path, map_location=VideoConfig.DEVICE)

        clean_state_dict = {}
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                name = k.replace('_orig_mod.', '').replace('module.', '')
                clean_state_dict[name] = v
            model.load_state_dict(clean_state_dict)
        else:
            raise ValueError("Downloaded file does not seem to be a state_dict.")

        model.to(VideoConfig.DEVICE)
        model.eval()
        print("Video model loaded successfully from downloaded Hub file.")
        return model, True, None

    except Exception as e:
        print(f"Error loading video model from Hub: {e}")
        st.error(f"Error loading video model: {e}")
        return None, False, str(e)

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    transform = transforms.Compose([
        transforms.Resize(VideoConfig.INPUT_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0).to(VideoConfig.DEVICE)
    return input_tensor

def predict_frame_emotion(model, frame):
    try:
        processed_input = preprocess_frame(frame)
        with torch.no_grad():
            output = model(processed_input)
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]
        predicted_emotion = VideoConfig.DEFAULT_CLASSES.get(predicted_class_index, f"Class {predicted_class_index}")
        return predicted_emotion, confidence
    except Exception as e:
        print(f"Error predicting frame: {e}")
        return "N/A", 0.0

def process_video_frames(video_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    temp_out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_out_file.name
    temp_out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    emotion_counts = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    progress_bar = st.progress(0, text="Analyzing video frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        predicted_emotion, confidence = predict_frame_emotion(model, frame)
        emotion_counts.append(predicted_emotion)

        text = f"Emotion: {predicted_emotion} ({confidence:.2%})"
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 10 + text_height), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

        progress = frame_count / total_frames if total_frames > 0 else 0
        progress_bar.progress(progress, text=f"Analyzing Frame {frame_count}/{total_frames} - Emotion: {predicted_emotion}")

    cap.release()
    out.release()
    progress_bar.empty()

    summary = Counter(emotion_counts)
    return output_path, summary