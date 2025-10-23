Catto-Lingo 😼: AI-Powered Multi-Modal Cat Emotion Analyzer

Project Summary

Catto-Lingo is an intelligent, multi-modal, deep-learning-based system designed to bridge the communication gap between humans and cats. By analyzing feline vocalizations (audio), visual cues (video/image), and textual descriptions of behavior, this application aims to translate complex signals into understandable emotions (e.g., angry, happy, relaxed, scared). This helps cat owners, veterinarians, and researchers better understand feline emotional states, contributing to improved animal welfare and care.

Technical Approach & Challenges

This project integrates three distinct AI models, each tackling a different modality of cat expression. The development involved significant challenges, particularly around model deployment due to size constraints and environment inconsistencies.

1. Audio Analysis (CNN - TensorFlow/Keras)

    Model: A Convolutional Neural Network (CNN) trained using TensorFlow/Keras.

    Input: Mel-spectrogram images generated from cat meow audio files (.wav, .mp3) using librosa.

    Output: Classifies audio into emotions like "Angry", "Happy", "Stressed", etc.

    Deployment: The trained model (.h5 file) is managed using Git LFS within the main repository due to its moderate size.

2. Video/Image Analysis (ResNet-50 - PyTorch)

    Model: A pre-trained ResNet-50 model, fine-tuned using PyTorch/Torchvision for cat emotion classification from visual data.

    Input: Processes video files (.mp4, etc.) frame-by-frame or accepts static images (.jpg, .png). Standard ImageNet pre-processing is applied.

    Output: Predicts the dominant emotion based on visual cues across video frames or from a single image. Outputs include the processed video with frame-level predictions and summary statistics.

    Challenge & Solution: The initial model was saved using a method (mlflow.pytorch.log_model) that implicitly used Python's pickle, leading to loading errors (invalid load key, 'V') during deployment due to environment/version mismatches.

        Resolution: The best model's weights were extracted and re-saved correctly using torch.save(model.state_dict(), ...). To overcome deployment size limits and ensure portability, the corrected model (.pth state dictionary) is now hosted on Hugging Face Hub https://huggingface.co/Nour87/cattolingo-video/tree/main. The application downloads it directly during runtime using huggingface_hub.hf_hub_download.

3. Text Analysis (RoBERTa-Large - PyTorch/Transformers)

    Model: A large-scale Transformer model (roberta-large) fine-tuned using PyTorch and the Hugging Face transformers library.

    Input: Textual descriptions of cat behavior (e.g., "the cat is purring loudly," "hissing with ears back").

    Output: Predicts the corresponding emotion with high accuracy (achieved 92% during testing).

    Challenge & Solution: The fine-tuned roberta-large model is very large (~1.5 GB), exceeding GitHub's and Git LFS's practical limits for smooth deployment.

        Resolution: The entire fine-tuned model (weights and configuration) is hosted on Hugging Face Hub https://huggingface.co/Nour87/cattolingo-nlp-roberta-large/tree/main. The application utilizes the transformers library's from_pretrained() function, which automatically downloads and caches the model from the Hub on first run, keeping the application repository lightweight.

Deployment Challenges & Solutions

Deploying these applications on free tiers (Streamlit Community Cloud, Hugging Face Spaces) presented several hurdles:

    Repository Cloning Errors: Initial attempts failed due to platform caching issues or confusion with repository names/forks. Resolved by creating clean, dedicated repositories and ensuring correct permissions/linking.

    inotify Limits (OSError: [Errno 24]): Streamlit's file watcher hit Linux system limits due to potentially large numbers of files (e.g., cached environments). Resolved by implementing a comprehensive .gitignore file to exclude unnecessary directories like venv and __pycache__.

    protobuf Conflicts (TypeError: Descriptors cannot be created...): An incompatibility between newer protobuf versions and Streamlit dependencies. Resolved by pinning a compatible version (protobuf==3.20.3) in requirements.txt.

    OpenCV Dependencies (ImportError: libGL.so.1): Required Linux system libraries were missing. Resolved by creating a packages.txt file (libgl1-mesa-glx, libglib2.0-0) for the deployment environment.

    Memory Limits (Resource limits exceeded): Particularly with the large NLP model, free tier RAM (often 1GB) proved insufficient for stable operation.

        Mitigation: Ensuring model loading uses caching (@st.cache_resource) helps, but roberta-large remains challenging on free tiers. Using Hugging Face Spaces sometimes offers slightly more lenient resource management compared to Streamlit Cloud's free tier. For guaranteed stability with large models, paid tiers or alternative platforms might be necessary.

Project Structure

    src/: Contains modular Python source code for each model's processing logic (e.g., nlp_processor.py, video_processor.py, audio_cnn_src.py). Code here is designed for reusability.

    streamlit/: Contains the individual Streamlit GUI application files for each model (e.g., cattolingo-NLP-gui.py).

    notebooks/: Jupyter Notebooks detailing the experimentation, training, and evaluation process for each model.

    models/: Stores smaller model artifacts (.h5, .pkl) tracked via Git LFS. Larger models (.pth, .safetensors) are linked via Hugging Face Hub.

    requirements.txt: Lists all necessary Python packages for the project.

    packages.txt: Lists system-level dependencies required for deployment (e.g., for OpenCV).

    .gitignore: Specifies intentionally untracked files/directories.

    .gitattributes: Configures Git LFS for specific file types.

(Note: Specific file/folder names might differ slightly across repositories)

Deployed Applications 🚀

Live versions of the individual Streamlit applications can be accessed via the links below. These represent the current "prototype" stage, demonstrating each model independently.
Model Type	Repository Source	Application Link	Status
Audio (CNN)	nour-wq277/... (Fork)	🔊 	✅ Deployed
Video (PyTorch)	nour-wq277/test	🎬 Launch App	✅ Deployed
Text (NLP)	nour-wq277/... (Fork)	💬 Launch App	✅ Deployed

(Note: The NLP model (roberta-large) hosted on Hugging Face Hub may take a significant amount of time to load on the first run as it downloads the ~1.5 GB model files.)

Repository Links

    Main Forked Repository (Used for Development/Deployments): https://github.com/NOUR-wq277/Cattolingo-Project (This repo contains the integrated codebase)

    Deployment Test Repository: https://github.com/NOUR-wq277/Test (Used specifically for troubleshooting video app deployment)

    Original Upstream Repository: nhahub/NHA-145

Setup and Installation (Local)

    Clone your primary repository:
    Bash

git clone https://github.com/NOUR-wq277/Cattolingo-Project.git
cd Cattolingo-Project

Install Git LFS: Make sure you have Git LFS installed (git lfs install).

Pull LFS Files: Download the smaller model files tracked by LFS.
Bash

git lfs pull

Set up Python Environment:
Bash

python -m venv venv
# Activate the environment (Windows example):
venv\Scripts\activate
# (Use source venv/bin/activate on Linux/Mac)
pip install -r requirements.txt

Run Applications:
Bash

    # Example: Run the NLP app
    streamlit run streamlit/cattolingo-NLP-gui.py

    (Note: Large models from Hugging Face Hub will be downloaded automatically on first run, requiring internet access.)

Future Work

This project serves as a strong foundation. Future development plans include:

    Backend API: Developing a robust backend using FastAPI to serve predictions from all models via REST endpoints.

    Integrated Frontend: Creating a single, unified web application (potentially using Streamlit or another framework) that consumes the API and offers a seamless multi-modal analysis experience.

    Advanced Features: Exploring the integration of a RAG (Retrieval-Augmented Generation) system or an AI Agent to provide richer context, explanations, or even conversational interaction based on the detected emotions.

    Model Optimization: Investigating techniques like model quantization or distillation to reduce the size and resource consumption of large models like roberta-large for more efficient deployment.
