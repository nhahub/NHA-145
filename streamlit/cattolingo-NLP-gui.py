import sys, os
import streamlit as st


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.cattolingo_nlp2_src import load_nlp_model, predict_emotion

st.set_page_config(
    page_title="Cat Emotion Detector 😺 (NLP)",
    layout="wide",
    page_icon="🐾"
)


st.markdown("<h1 style='text-align:center; color:#FF6B6B;'>🐱 Cat Emotion Detector (NLP)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4ECDC4;'>Detect your cat's emotion from text — RoBERTa Engine 🤖</p>", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    with st.spinner("Loading AI model (RoBERTa)..."):
        model, tokenizer = load_nlp_model()
        return model, tokenizer

model, tokenizer = get_model()



with st.sidebar:
    st.image("https://images.unsplash.com/photo-1574144611937-0df059b5ef3e?w=400", use_column_width=True)
    st.markdown("### ⌨️ Enter Text")
    
   
    text = st.text_area("Enter text to analyze (e.g., 'the cat is purring')", height=150)
    
  
    analyze_button = st.button("🔮 Predict Emotion", use_container_width=True)
    
    st.info("🤖 Model: RoBERTa (Large) trained on emotion text.")



if not model or not tokenizer:
    st.error("❌ Model not loaded properly. Please check your model path.")
elif not text:
    st.info("⬆️ Please enter some text in the sidebar to start analysis.")
else:

    st.markdown("### 💬 Text to Analyze:")
    st.info(text)

    if analyze_button:
        with st.spinner("Analyzing text... ⏳"):
            try:
             
                prediction = predict_emotion(text, model, tokenizer)
                
                st.markdown("### 🎯 Prediction Result")
                st.success(f"**Emotion:** {prediction}")
            
                emotion_emoji = {
                    'angry': '😡', 'disgusted': '🤢', 'happy': '😄', 'normal': '😐',
                    'relaxed': '😌', 'sad': '😢', 'scared': '😨', 'surprised': '😮', 'uncomfortable': '😖'
                }
                st.markdown(f"<h1 style='text-align:center; font-size: 5rem;'>{emotion_emoji.get(prediction, '❓')}</h1>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit and Transformers — RoBERTa Engine</p>",
    unsafe_allow_html=True
)