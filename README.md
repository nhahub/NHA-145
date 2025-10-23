# 🐱 CattoLingo: Multi-Modal Cat Emotion Analyzer

<p align="center">
  <img src="D:\NHA-145\Screenshot 2025-10-23 215442.png" alt="CattoLingo Demo" width="800"/>
</p>

---

## 🚀 Project Summary

CattoLingo is an intelligent, multi-modal, deep-learning-based system designed to bridge the communication gap between humans and cats.  
By analyzing feline vocalizations (**audio**), visual cues (**video/image**), and textual descriptions of behavior (**text**), this application translates signals into understandable emotions like **angry**, **happy**, **relaxed**, or **stressed** — helping cat owners and researchers better understand feline emotional states.

---

## 🧠 Technical Overview

CattoLingo integrates **three independent AI models**, each focused on a different modality.  
The main challenge was handling **large model sizes**, **environment mismatches**, and **limited free-tier deployment resources** — all addressed through optimized packaging, model hosting, and Hugging Face integration.

---

## 🐾 Model Breakdown

| Modality | Model Type | Framework | Deployment | Hugging Face |
|-----------|-------------|------------|-------------|---------------|
| 🔊 Audio | CNN on Mel-Spectrograms | TensorFlow/Keras | Streamlit App | [cattolingo-audio-cnn](https://huggingface.co/Nour87/cattolingo-audio-cnn/tree/main) |
| 🎬 Video | ResNet-50 | PyTorch | Streamlit App | [cattolingo-video](https://huggingface.co/Nour87/cattolingo-video/tree/main) |
| 💬 Text | RoBERTa-Large | PyTorch/Transformers | Streamlit App | [cattolingo-nlp-roberta-large](https://huggingface.co/Nour87/cattolingo-nlp-roberta-large/tree/main) |

---

## ⚙️ Deployment Challenges & Solutions

- **Model Size Limits:**  
  → Hosted large `.pth` and `.safetensors` files on Hugging Face Hub using `hf_hub_download`.

- **Streamlit Crashes (inotify / memory errors):**  
  → Added `.gitignore` for heavy folders and applied `@st.cache_resource`.

- **Protobuf Conflicts:**  
  → Pinned compatible version `protobuf==3.20.3`.

- **OpenCV Missing libs:**  
  → Fixed via `packages.txt` → `libgl1-mesa-glx`, `libglib2.0-0`.

---

## 🗂️ Project Structure
📦 Cattolingo-Project/
 ┣ 📂 src/                
 ┣ 📂 streamlit/           
 ┣ 📂 notebooks/          
 ┣ 📂 models/              
 ┣ 📄 requirements.txt
 ┣ 📄 packages.txt
 ┣ 📄 .gitignore
 ┣ 📄 README.md

---

## 🌐 Live Applications

| Model | Repo | Live Demo | Status |
|--------|-------|------------|--------|
| 🔊 Audio (CNN) | [Test3](https://github.com/NOUR-wq277/Test3/tree/main) | [Streamlit App](https://nkpwzdsc2szjybzndwr4hs.streamlit.app/) | ✅ Deployed |
| 🎬 Video (ResNet-50) | [Test](https://github.com/NOUR-wq277/Test) | [Streamlit App](https://gyj6tclynmbfe9bove5eno.streamlit.app/) | ✅ Deployed |
| 💬 Text (RoBERTa-Large) | [Test2](https://github.com/NOUR-wq277/Test2) | [Streamlit App](https://ufotvesktp5ai2xzcrdkpn.streamlit.app/) | ✅ Deployed |

---

## 🧩 Future Work

- 🔗 Unified multi-modal app combining all models  
- ⚡ FastAPI backend for REST-based predictions  
- 💬 Integrated chat agent for emotion interpretation  
- 🧠 Model quantization / distillation for lightweight deployment  

---

## 📜 References

- TensorFlow/Keras, PyTorch, Hugging Face Transformers  
- OpenCV, librosa, Streamlit, Plotly  
- Dataset curation and model weights hosted on Hugging Face Hub  

---

👨‍💻 **Developer:** [Nour Ahmed](https://github.com/NOUR-wq277)  
📦 **Main Repo:** [Cattolingo-Project](https://github.com/NOUR-wq277/Cattolingo-Project)


