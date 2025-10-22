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
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cattolingo_Emotion_Detector_src import (
    load_classification_model,
    predict_frame_emotion,
    process_video_frames,
    VideoConfig
)

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
    st.session_state.class_names = VideoConfig.DEFAULT_CLASSES.copy()
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = VideoConfig.NUM_FIXED_CLASSES

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

    with st.sidebar:
        st.header("⚙ Settings")
        st.markdown(f"*الجهاز المستخدم:* <span style='color:#4ECDC4;'>{VideoConfig.DEVICE.type.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"*هيكل النموذج:* <span style='color:#FF6B6B;'>ResNet-50</span>", unsafe_allow_html=True)
        st.markdown("---")

        if st.button("🔄 Load Model from Hub", type="primary"):
            with st.spinner("Loading model from Hub..."):
                model, success, error = load_classification_model()
                if success:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully from Hub!")
                    st.info(f"📋 Model is configured for {st.session_state.num_classes} emotions.")
                else:
                    st.error(f"❌ Error loading model: {error}")

        st.markdown("---")
        st.subheader("🏷 Emotion Labels")
        st.markdown(f"*Classes ({st.session_state.num_classes}):*")
        st.code(', '.join(st.session_state.class_names.values()), language='text')
        st.markdown("---")

    if not st.session_state.model_loaded:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.warning("⚠ يرجى تحميل النموذج من الشريط الجانبي (Sidebar) للبدء!")
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

        if st.button("🔮 Analyze Cat Emotions", type="primary", use_container_width=True):
            if st.session_state.model:
                with st.spinner("Analyzing video... This might take a while..."):
                    try:
                        output_video_path, emotions_summary_counter = process_video_frames(
                            video_path,
                            st.session_state.model
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
                            if total_frames > 0:
                                df_summary['Percentage'] = (df_summary['Frame Count'] / total_frames)
                            else:
                                df_summary['Percentage'] = 0.0

                            if not df_summary.empty:
                                dominant = df_summary.loc[df_summary['Frame Count'].idxmax()]
                                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                st.markdown(f"## 🏆 *{dominant['Emotion']}* (Dominant Emotion)")
                                st.markdown(f"### Appeared in *{dominant['Frame Count']}* frames, or *{dominant.get('Percentage', 0):.1%}* of the video.")
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
                                    'confidence': dominant.get('Percentage', 0),
                                    'image': uploaded_file.name
                                })
                            else:
                                st.warning("No emotions detected in the video frames.")

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
                with col1: st.write(f"*Video:* {pred['image']}")
                with col2: st.write(f"*Dominant Emotion:* {pred.get('emotion', 'N/A')}")
                with col3: st.write(f"*Percentage:* {pred['confidence']:.2%}")
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
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()