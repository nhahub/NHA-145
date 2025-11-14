# 🐱 Catto-Lingo: AI-Powered Multi-Modal Cat Emotion Analyzer

<div align="center">
  <img src="D:\NHA-145\Screenshot 2025-10-23 215442.png" alt="CattoLingo Demo" width="800"/>
  <br>
  <h3><i>"Bridging the communication gap between you and your cat."</i></h3>
</div>

---

## 🚀 Project Overview

**Catto-Lingo** is an advanced Artificial Intelligence system designed to interpret feline emotions through multiple modalities: **Audio** (Vocalizations), **Visuals** (Facial Expressions/Body Language), and **Textual Descriptions** (Behavioral Context).

This project represents a complete engineering journey, evolving from experimental **Deep Learning research notebooks** into a robust, **Full-Stack Microservices Architecture** powered by Cloud Computing, Docker, and Generative AI Agents.

---

# 🌟 The Final Product (Full-Stack Deployment)

In the final phase, we integrated all independent models into a unified, production-grade web application.


---

# 🧪 Phase 1: Research & Prototypes (Individual Models)

Before system integration, each modality was developed, trained, and deployed independently to validate model performance. You can still access these individual prototypes below.

### 🔗 Prototype Deployments

| Modality | Model Architecture | Framework | Live Demo |
| :--- | :--- | :--- | :--- |
| **🔊 Audio Analysis** | **Custom CNN** (Mel-Spectrograms) | TensorFlow/Keras | [👉 Try Audio App](https://nkpwzdsc2szjybzndwr4hs.streamlit.app/) |
| **🎬 Video Analysis** | **ResNet-50** (Fine-Tuned) | PyTorch | [👉 Try Video App](https://gyj6tclynmbfe9bove5eno.streamlit.app/) |
| **💬 Text Analysis** | **RoBERTa-Large** (Transformer) | Hugging Face | [👉 Try Text App](https://ufotvesktp5ai2xzcrdkpn.streamlit.app/) |

---
### 🔗 Live Production System

| Component | Technology Stack | Status | Access Link |
| :--- | :--- | :---: | :--- |
| **🌐 Web Application** | **Frontend:** HTML5, CSS3, JS (Netlify) | ✅ Live | [**🚀 Launch Catto-Lingo**](https://brilliant-starburst-694b2f.netlify.app/) |
| **🧠 Backend API** | **Backend:** FastAPI, Docker (HF Spaces) | ✅ Live | [**📄 API Documentation**](https://huggingface.co/spaces/Nour87/cattolingo-api/tree/main) |
| **🗄️ Database** | **DB:** PostgreSQL (Supabase) | ✅ Active | *Cloud Connected* |


## 🧠 Technical Deep Dive & Engineering Challenges

### 1. Audio Analysis (CNN) 🎤
* **Objective:** Classify cat meows into emotions (Angry, Happy, Sad, Stressed, Resting).
* **Methodology:** Raw audio signals were converted into **Mel-Spectrograms** (visual representations of sound). A custom Convolutional Neural Network (CNN) was trained on these images to extract time-frequency features.
* **Preprocessing:** Utilized `librosa` for signal processing and normalization to handle variable audio lengths and background noise.

### 2. Video/Image Analysis (ResNet-50) 🎬
* **Objective:** Detect dominant emotions from static images or video streams.
* **Methodology:** Transfer Learning using a **ResNet-50** architecture pre-trained on ImageNet and fine-tuned on a curated dataset of cat facial expressions.
* **Engineering Challenge:** The initial model serialization using `mlflow` caused `pickle` compatibility issues (`invalid load key 'V'`) when deployed to cloud environments with different PyTorch versions.
* **Solution:** We refactored the deployment pipeline to extract the model's **`state_dict`** (weights only) and rebuilt the model architecture dynamically during runtime, ensuring cross-platform compatibility.

### 3. Text Analysis (RoBERTa-Large) 💬
* **Objective:** Understand complex behavioral descriptions (e.g., *"The cat is hissing with ears flattened"*).
* **Methodology:** Fine-tuning the **RoBERTa-Large** transformer model for sequence classification. Achieved an accuracy of **92%**.
* **Engineering Challenge:** The model size (**~1.5 GB**) exceeded the hard limits of GitHub repositories and standard Git LFS quotas, causing deployment failures.
* **Solution:** We implemented a **Hybrid Storage Architecture**. The heavy model weights were hosted on **Hugging Face Hub**, while the inference code resides on the API server. The system utilizes a "Lazy Loading" strategy to fetch and cache the model only upon initialization to optimize memory usage.

---

## 🏗️ System Architecture (Phase 2)

To move from prototypes to production, we re-architected the solution:

1.  **Backend (FastAPI + Docker):** * Developed a unified REST API using **FastAPI**.
    * Containerized the application using **Docker** to manage system-level dependencies (like `libgl1` for OpenCV).
    * Deployed on **Hugging Face Spaces** to leverage high-availability infrastructure.
2.  **Database (PostgreSQL):**
    * Integrated **Supabase** (Managed PostgreSQL) to store user profiles and prediction history.
    * Implemented **JWT Authentication** for secure user access.
3.  **AI Advisor Agent:**
    * Integrated **Google Gemini API** to act as a veterinary advisor.
    * The agent analyzes the predicted emotion + historical data from the database to generate personalized, empathetic advice for the pet owner.
4.  **Frontend:**
    * A responsive web interface deployed on **Netlify** that communicates asynchronously with the backend.

---

## 🛠️ Tech Stack

* **AI/ML:** PyTorch, TensorFlow, Keras, Transformers, Scikit-learn, OpenCV, Librosa.
* **Backend:** Python, FastAPI, Uvicorn, SQLAlchemy, Pydantic.
* **DevOps:** Docker, Git LFS, Hugging Face Spaces.
* **Database:** PostgreSQL, Supabase.
* **Frontend:** HTML5, CSS3, JavaScript.

---

## 👥 The Team

This project was brought to life by the dedicated efforts of:

* **Nour Ahmed**
* **Abdalrhman Elsaid**
* **Omnia Elmetwally**
* **Ziad Sakr**
* **Abdalrhman Ibrahim**
* **Belal Mahmoud**


---
