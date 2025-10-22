import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import tempfile
from collections import Counter
import pandas as pd
import json
import plotly.express as px
import torch
import torch.nn as nn
from torchvision import models
import mlflow
import mlflow.pytorch
from torchvision import transforms # Added this import back

base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DEFAULT_PATH = os.path.join(base_dir, '..', 'models', 'cattolingo-Emotion_Detector-model.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASSES = {
    0: 'angry',
    1: 'disgusted',
    2: 'happy',
    3: 'normal',
    4: 'relaxed',
    5: 'sad',
    6: 'scared',
    7: 'surprised',
    8: 'uncomfortable'
}
NUM_FIXED_CLASSES = len(DEFAULT_CLASSES)
INPUT_IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Cat Emotion Video Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #FF6B6B; text-align: center; font-weight: bold; margin-bottom: 1rem; }
    .sub-header { font-size: 1.5rem; color: #4ECDC4; text-align: center; margin-bottom: 2rem; }
    .prediction-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #FF6B6B; }
    .info-box { background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = DEFAULT_CLASSES.copy()
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = NUM_FIXED_CLASSES

@st.cache_resource
def load_classification_model(model_path, num_classes):
    try:
        if not os.path.exists(model_path):
            return None, False, f"Model file not found: {model_path}"

        try:
            model = mlflow.pytorch.load_model(model_uri=model_path, map_location=DEVICE)
            print("Model loaded successfully using mlflow.pytorch.load_model.")

        except Exception as mlflow_e:
            print(f"mlflow.pytorch.load_model failed ({mlflow_e}), trying torch.load with fallback...")
            try:
                model_structure = models.resnet50(pretrained=False)
                num_features = model_structure.fc.in_features
                model_structure.fc = nn.Linear(num_features, num_classes)

                state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)

                clean_state_dict = {}
                if isinstance(state_dict, dict):
                    for k, v in state_dict.items():
                        name = k.replace('_orig_mod.', '').replace('module.', '')
                        clean_state_dict[name] = v
                    model_structure.load_state_dict(clean_state_dict)
                    model = model_structure
                    print("Fallback successful: Loaded state_dict using torch.load.")
                elif isinstance(state_dict, nn.Module):
                     model = state_dict
                     print("Fallback successful: Loaded full model using torch.load.")
                else:
                     raise ValueError("Loaded object is not a state_dict or nn.Module")

            except Exception as torch_e:
                 print(f"torch.load fallback also failed: {torch_e}")
                 return None, False, f"Failed to load model using both MLflow and torch.load. Original MLflow error: {mlflow_e}; Torch error: {torch_e}"

        model.to(DEVICE)
        model.eval()
        return model, True, None

    except Exception as e:
        return None, False, str(e)

def preprocess_frame(frame, target_size):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    return input_tensor

def predict_emotion(model, frame, class_names):
    try:
        processed_input = preprocess_frame(frame, INPUT_IMAGE_SIZE)

        with torch.no_grad():
            output = model(processed_input)

        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]
        predicted_emotion = class_names.get(predicted_class_index, f"Class {predicted_class_index}")

        return predicted_emotion, confidence, None
    except Exception as e:
        return "N/A", 0.0, None


def process_video(video_path, model, class_names):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_out_file.name
    temp_out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    emotion_counts = []

    my_bar = st.progress(0, text="Operation in progress. Please wait.")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        predicted_emotion, confidence, _ = predict_emotion(model, frame, class_names)
        emotion_counts.append(predicted_emotion)

        text = f"Emotion: {predicted_emotion} ({confidence:.2%})"
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline), (0, 0, 0), -1)

        cv2.putText(frame, text, (15, 10 + text_height), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

        progress = frame_count / total_frames if total_frames > 0 else 0
        my_bar.progress(progress, text=f"Analyzing Frame {frame_count}/{total_frames} - Current Emotion: {predicted_emotion}")

    cap.release()
    out.release()
    my_bar.empty()

    summary = Counter(emotion_counts)

    return output_path, summary

def create_probability_chart(df_summary):
    fig = px.bar(df_summary.sort_values(by='Frame Count', ascending=True),
                 x='Frame Count', y='Emotion', orientation='h',
                 title="Emotion Distribution (Total Frames Analyzed)",
                 color='Frame Count', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Frame Count", yaxis_title="Cat Emotion", height=400, showlegend=False)
    return fig

def create_donut_chart(df_summary):
    fig = px.pie(df_summary, values='Percentage', names='Emotion',
                 title="Overall Percentage Breakdown of Emotions", hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Mint)
    fig.update_layout(height=300, showlegend=True)
    return fig

def main():
    st.markdown('<p class="main-header">🎬 Cat Emotion Video Detector (PyTorch)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Frame-by-Frame Analysis Powered by Deep Learning</p>', unsafe_allow_html=True)

    model_default = MODEL_DEFAULT_PATH

    with st.sidebar:
        st.header("⚙ Settings")

        model_path_input = st.text_input(
            "Model Path (.pth)",
            value=model_default,
            help="Path to your trained PyTorch model file (.pth)"
        )

        st.markdown(f"*الجهاز المستخدم:* <span style='color:#4ECDC4;'>{DEVICE.type.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"*هيكل النموذج:* <span style='color:#FF6B6B;'>ResNet-50</span>", unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model and classes..."):
                current_num_classes = st.session_state.num_classes
                model, success, error = load_classification_model(model_path_input, current_num_classes)

                if success:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    st.info(f"📋 Model is configured for {current_num_classes} fixed emotions: {', '.join(st.session_state.class_names.values())}")
                else:
                    st.error(f"❌ Error loading model: {error}")

        st.markdown("---")

        st.subheader("🏷 Emotion Labels (Fixed)")
        st.markdown(f"*Fixed Classes ({NUM_FIXED_CLASSES}):*")
        st.code(', '.join(st.session_state.class_names.values()), language='text')

        st.markdown("---")

    if not st.session_state.model_loaded:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.warning("⚠ يرجى تحميل النموذج من الشريط الجانبي (Sidebar) للبدء! (Model Path)")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("---")
    st.subheader("📤 Upload Cat Video")

    uploaded_file = st.file_uploader(
        "Choose a video file...",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video of a cat for emotion detection"
    )

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.type.split('/')[-1]}")
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        temp_file.close()

        st.markdown("### 🎬 Input Video")
        st.video(video_path)

        st.markdown("---")

        if st.button("🔮 Analyze Cat Emotions in Video", type="primary", use_container_width=True):
            if st.session_state.model and st.session_state.class_names:
                with st.spinner("Analyzing video... This might take a while..."):
                    try:
                        output_video_path, emotions_summary_counter = process_video(
                            video_path,
                            st.session_state.model,
                            st.session_state.class_names
                        )

                        os.unlink(video_path)

                        if output_video_path and emotions_summary_counter:
                            st.subheader("🎯 Processed Video with Predictions")
                            st.video(output_video_path)
                            os.unlink(output_video_path)

                            st.markdown("---")
                            st.subheader("📊 Video Analysis Summary")

                            df_summary = pd.DataFrame(emotions_summary_counter.items(), columns=['Emotion', 'Frame Count'])
                            total_frames = df_summary['Frame Count'].sum()
                            df_summary['Percentage'] = (df_summary['Frame Count'] / total_frames)

                            dominant = df_summary.loc[df_summary['Frame Count'].idxmax()]

                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown(f"## 🏆 *{dominant['Emotion']}* (Dominant Emotion)")
                            st.markdown(f"### Appeared in *{dominant['Frame Count']}* frames, or *{dominant['Percentage']:.1%}* of the video.")
                            st.markdown('</div>', unsafe_allow_html=True)

                            tab1, tab2 = st.tabs(["📊 Frame Count Distribution", "🎯 Overall Breakdown"])

                            with tab1:
                                fig_bar = create_probability_chart(df_summary)
                                st.plotly_chart(fig_bar, use_container_width=True)

                            with tab2:
                                fig_donut = create_donut_chart(df_summary)
                                st.plotly_chart(fig_donut, use_container_width=True)

                            st.session_state.prediction_history.append({
                                'emotion': dominant['Emotion'],
                                'confidence': dominant['Percentage'],
                                'image': uploaded_file.name
                            })

                        else:
                            st.error("❌ Video processing failed or no frames were analyzed.")

                    except Exception as e:
                        st.error(f"❌ Error during video analysis: {str(e)}")
            else:
                st.warning("⚠ Please load a model first!")

    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📜 Analysis History")

        with st.expander("View All Summaries", expanded=False):
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"*Video:* {pred['image']}")
                with col2:
                    st.write(f"*Dominant Emotion:* {pred.get('emotion', 'N/A')}")
                with col3:
                    st.write(f"*Percentage:* {pred['confidence']:.2%}")
                st.markdown("---")

        if st.button("🗑 Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Built with ❤ using Streamlit, OpenCV, and PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()