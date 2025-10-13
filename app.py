"""
Streamlit GUI for Cat Emotion Detector - Video Analysis
PyTorch Edition: Implemented with ResNet-50 architecture.
"""

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

# NEW IMPORTS FOR PYTORCH
import torch
import torch.nn as nn
from torchvision import transforms, models
# يمكن استخدام ResNet50_Weights إذا كان إصدار PyTorch لديك حديثاً ويدعمها
# من أجل تبسيط النشر، سنستخدم طريقة التحميل القديمة الموثوقة (models.resnet50(pretrained=True))

# ============================================================================
# --- USER CONFIGURATION (Required) ---
# ============================================================================

# تم تحديث المسارات بناءً على ما أرسلته (استخدام سلاش أمامي أو سلاش مزدوج للخلف لتجنب مشاكل المسار)
# يتم استخدام المسار النسبي/الافتراضي ليتناسب مع بيئة Streamlit عند التشغيل
MODEL_DEFAULT_PATH = "./models/Cat Emotion Detector model.pth" 
TRAIN_DATA_PATH = "./cats_dataset_balanced/train"

# الإعدادات المكتشفة من الكود الخاص بك
INPUT_IMAGE_SIZE = (224, 224) # Standard input size for ResNet-50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PAGE CONFIGURATION AND STYLING (Preserved Design)
# ============================================================================

st.set_page_config(
    page_title="Cat Emotion Video Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS STYLING (Preserved)
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    .info-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = 9 # Set to 9 based on notebook analysis


# ============================================================================
# HELPER FUNCTIONS (PyTorch Logic - ResNet-50 Implementation)
# ============================================================================

# ابحث عن هذه الدالة واستبدلها بالكامل:

@st.cache_resource
def load_classification_model(model_path, num_classes):
    """
    Load the trained PyTorch model (ResNet-50 structure) with caching.
    This function replicates the model structure found in Cat-Emotion-Detector.ipynb.
    """
    try:
        if not os.path.exists(model_path):
            return None, False, f"Model file not found: {model_path}"
        
        # 1. Instantiate the ResNet-50 structure
        # (استخدام pretrained=True لتحميل الأوزان الجاهزة لـ ImageNet كبداية)
        model = models.resnet50(pretrained=True) 
        
        # تجميد الطبقات (اختياري، لكن يفضل لتقليل الذاكرة)
        for param in model.parameters():
             param.requires_grad = False
             
        # الحصول على عدد الميزات لطبقة التصنيف النهائية
        num_features = model.fc.in_features
        
        # 2. استبدال طبقة التصنيف النهائية لتتناسب مع عدد الفئات (9)
        model.fc = nn.Linear(num_features, num_classes)
        
        # 3. تحميل الأوزان من ملف .pth
        state_dict_or_model = torch.load(model_path, map_location=DEVICE)
        
        # التحقق مما إذا كان الملف يحتوي على state_dict مباشرة أو داخل مفتاح
        if isinstance(state_dict_or_model, dict):
            # محاولة التحميل من المفاتيح الشائعة أولاً
            if 'model_state_dict' in state_dict_or_model:
                 model.load_state_dict(state_dict_or_model['model_state_dict'])
            elif 'state_dict' in state_dict_or_model:
                 model.load_state_dict(state_dict_or_model['state_dict'])
            else:
                 # إذا كان state_dict هو القاموس بأكمله
                 model.load_state_dict(state_dict_or_model)
        else:
            # إذا كان الملف المحفوظ يحتوي على كائن النموذج بالكامل
            # في هذه الحالة، سنستخدم النموذج المحفوظ مباشرة (لحل مشكلة iterable)
            # ولكن يجب أن نتأكد من أنه من نوع nn.Module
            if isinstance(state_dict_or_model, nn.Module):
                 model = state_dict_or_model
            else:
                 # إذا لم يكن قاموساً أو نموذجاً، قد يكون هذا خطأ في صيغة الحفظ
                 return None, False, "Error: The .pth file content is neither a state dictionary nor a PyTorch model instance."


        model.to(DEVICE)
        model.eval() # تعيين النموذج لوضع التقييم
        
        return model, True, None
    except Exception as e:
        # إرجاع أي خطأ آخر يحدث أثناء عملية التحميل
        return None, False, str(e)


def get_class_names_from_directory(train_dir):
    """Extract class names from training directory structure"""
    try:
        # Check if the directory exists using the Streamlit-friendly relative path first
        if not os.path.exists(train_dir):
             # Fallback to absolute path check if necessary, though it complicates deployment
             # For now, we rely on the Streamlit path structure
             return None, 0

        classes = sorted([d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))])
        return {i: name for i, name in enumerate(classes)}, len(classes)
    except Exception as e:
        # print(f"Error reading classes: {e}")
        return None, 0

def preprocess_frame(frame, target_size):
    """
    Preprocess a single frame using PyTorch transforms.
    Normalization uses ImageNet standards, matching the notebook.
    """
    
    # 1. Convert OpenCV BGR frame to PIL RGB Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # 2. Define standard PyTorch transformations (matching val_test_transforms from notebook)
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(), # Converts to C x H x W and scales to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Apply transformation
    input_tensor = transform(img)
    
    # 4. Add batch dimension and send to device
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    return input_tensor

def predict_emotion(model, frame, class_names):
    """Perform prediction on a single frame using PyTorch logic"""
    try:
        # 1. Preprocess frame
        processed_input = preprocess_frame(frame, INPUT_IMAGE_SIZE)
        
        # 2. Predict (Disable gradient calculations for inference)
        with torch.no_grad():
            output = model(processed_input)
        
        # 3. Apply Softmax to get probabilities (needed if model output is logits)
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # 4. Get results
        predicted_class_index = np.argmax(probabilities)
        confidence = probabilities[predicted_class_index]
        predicted_emotion = class_names.get(predicted_class_index, f"Class {predicted_class_index}")
        
        return predicted_emotion, confidence, None
    except Exception as e:
        # print(f"Prediction Error: {e}")
        return "N/A", 0.0, None


def process_video(video_path, model, class_names):
    """
    Reads a video, applies the model on each frame, adds the prediction, 
    and saves the processed video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Use a temporary file path
    temp_out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_out_file.name
    temp_out_file.close()

    # Use mp4v codec for broad compatibility
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

        # --- Prediction on Frame ---
        predicted_emotion, confidence, _ = predict_emotion(model, frame, class_names)
        emotion_counts.append(predicted_emotion)
        
        # --- Annotation on Video ---
        text = f"Emotion: {predicted_emotion} ({confidence:.2%})"
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw a black background box for better text visibility
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline), (0, 0, 0), -1)
        
        # Draw white text
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
    """Create a horizontal bar chart for frame counts by emotion"""
    fig = px.bar(df_summary.sort_values(by='Frame Count', ascending=True), 
                 x='Frame Count', y='Emotion', orientation='h',
                 title="Emotion Distribution (Total Frames Analyzed)",
                 color='Frame Count', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Frame Count", yaxis_title="Cat Emotion", height=400, showlegend=False)
    return fig

def create_donut_chart(df_summary):
    """Create a donut chart showing percentage breakdown of emotions"""
    fig = px.pie(df_summary, values='Percentage', names='Emotion', 
                 title="Overall Percentage Breakdown of Emotions", hole=0.4,
                 color_discrete_sequence=px.colors.sequential.Mint)
    fig.update_layout(height=300, showlegend=True)
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    st.markdown('<p class="main-header">🎬 Cat Emotion Video Detector (PyTorch)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Frame-by-Frame Analysis Powered by Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # --- Model Path ---
        # Note: The Streamlit app runs from its own directory, so relative paths are often preferred.
        model_path_input = st.text_input(
            "Model Path (.pth)",
            value=MODEL_DEFAULT_PATH,
            help="Path to your trained PyTorch model file (.pth)"
        )
        # --- Data Path ---
        train_data_path_input = st.text_input(
            "Training Data Path (for class names)",
            value=TRAIN_DATA_PATH,
            help="Path to the parent directory of your class folders (e.g., .../train)"
        )
        st.markdown(f"**الجهاز المستخدم:** <span style='color:#4ECDC4;'>{DEVICE.type.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**هيكل النموذج:** <span style='color:#FF6B6B;'>ResNet-50</span>", unsafe_allow_html=True)

        st.markdown("---")
        
        # Load model button
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model and classes..."):
                
                # 1. Get class names and count first
                class_names, num_classes = get_class_names_from_directory(train_data_path_input)
                
                if num_classes == 0:
                    st.session_state.num_classes = 9 # Enforce 9 based on your notebook
                    st.warning(f"⚠️ لم يتم اكتشاف الفئات آلياً. سيتم استخدام العدد الثابت: 9")
                else:
                    st.session_state.num_classes = num_classes
                    st.session_state.class_names = class_names
                
                
                # 2. Load model using the detected/default class count
                model, success, error = load_classification_model(model_path_input, st.session_state.num_classes)
                
                if success:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    
                    if class_names:
                        st.session_state.class_names = class_names
                        st.info(f"📋 Detected {len(class_names)} emotions from data directory: {', '.join(class_names.values())}")
                    else:
                        st.session_state.class_names = {i: f"Class {i}" for i in range(st.session_state.num_classes)}
                        st.warning("⚠️ Could not auto-detect emotions. Please edit labels below.")
                        
                else:
                    st.error(f"❌ Error loading model: {error}")
        
        st.markdown("---")
        
        # Class names editor (Preserved)
        st.subheader("🏷️ Edit Emotion Labels")
        if st.session_state.class_names:
            num_classes = len(st.session_state.class_names)
            
            new_class_names = {}
            for idx in range(num_classes):
                current_name = st.session_state.class_names.get(idx, f"Class {idx}")
                new_name = st.text_input(f"Class {idx}", value=current_name, key=f"class_{idx}")
                new_class_names[idx] = new_name
            
            st.session_state.class_names = new_class_names

        else:
            # Set default 9 labels from notebook analysis if not loaded
            if st.session_state.num_classes == 9:
                 default_classes = ['angry', 'disgusted', 'happy', 'normal', 'relaxed', 'sad', 'scared', 'surprised', 'uncomfortable']
                 st.session_state.class_names = {i: name for i, name in enumerate(default_classes)}
            
            # Re-render the editor after setting defaults
            if st.session_state.class_names:
                st.subheader("🏷️ Edit Emotion Labels")
                num_classes = len(st.session_state.class_names)
                new_class_names = {}
                for idx in range(num_classes):
                    current_name = st.session_state.class_names.get(idx, f"Class {idx}")
                    new_name = st.text_input(f"Class {idx}", value=current_name, key=f"class_{idx}")
                    new_class_names[idx] = new_name
                st.session_state.class_names = new_class_names
            else:
                st.warning("Load the model first to detect/set emotion labels.")
        
        st.markdown("---")
        
    
    # Main content area
    if not st.session_state.model_loaded:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.warning("⚠️ يرجى تحميل النموذج وأسماء الفئات من الشريط الجانبي (Sidebar) للبدء!")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # File uploader (Video)
    st.markdown("---")
    st.subheader("📤 Upload Cat Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file...",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video of a cat for emotion detection"
    )
    
    # Prediction section (Video Logic)
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
                        # PROCESS THE VIDEO
                        output_video_path, emotions_summary_counter = process_video(
                            video_path,
                            st.session_state.model,
                            st.session_state.class_names
                        )
                        
                        os.unlink(video_path)

                        if output_video_path and emotions_summary_counter:
                            
                            st.subheader("🎯 Processed Video with Predictions")
                            st.video(output_video_path)
                            # Ensure the file is unlinked after display
                            os.unlink(output_video_path) 
                            
                            # --- Display Summary Charts ---
                            st.markdown("---")
                            st.subheader("📊 Video Analysis Summary")
                            
                            df_summary = pd.DataFrame(emotions_summary_counter.items(), columns=['Emotion', 'Frame Count'])
                            total_frames = df_summary['Frame Count'].sum()
                            df_summary['Percentage'] = (df_summary['Frame Count'] / total_frames)
                            
                            dominant = df_summary.loc[df_summary['Frame Count'].idxmax()]
                            
                            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                            st.markdown(f"## 🏆 **{dominant['Emotion']}** (Dominant Emotion)")
                            st.markdown(f"### Appeared in **{dominant['Frame Count']}** frames, or **{dominant['Percentage']:.1%}** of the video.")
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
                st.warning("⚠️ Please load a model and class names first!")

    
    # Prediction history (Preserved)
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📜 Analysis History")
        
        with st.expander("View All Summaries", expanded=False):
            # Show last 10 predictions
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-10:])):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**Video:** {pred['image']}")
                with col2:
                    st.write(f"**Dominant Emotion:** {pred.get('emotion', 'N/A')}")
                with col3:
                    st.write(f"**Percentage:** {pred['confidence']:.2%}")
                st.markdown("---")
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Footer (Preserved)
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Built with ❤️ using Streamlit, OpenCV, and PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()