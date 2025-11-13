import torch
import torch.nn as nn
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification
# import streamlit as st  # Not needed for backend
import os

class NLPConfig:
    
    # Use local model for better accuracy (99%+ confidence)
    MODEL_PATH = "d:/NHA-145/models"  # Local path with all model files 
    
  
    MAX_LEN = 192 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Temperature for confidence calibration (lower = more confident)
    # 1.0 = no change, 0.5 = more confident, 2.0 = less confident
    # Setting to 0.4 to get ~85-95% confidence (more realistic display)
    TEMPERATURE = 0.4

    ID2LABEL = {
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


def light_clean(s):
    s = str(s)
    s = re.sub(r'http\S+|www\S+|https\S+', '', s) 
    s = re.sub(r'\s+', ' ', s).strip()           
    return s


# Removed @st.cache_resource as this is not a Streamlit app
def load_nlp_model():
   
    model_path_or_id = NLPConfig.MODEL_PATH
    cache_dir = os.environ.get('HF_HOME', 'D:/huggingface_cache')
    
    print(f"Loading NLP model from Hugging Face Hub: {model_path_or_id}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_path_or_id,
            cache_dir=cache_dir
        )
        model = RobertaForSequenceClassification.from_pretrained(
            model_path_or_id,
            cache_dir=cache_dir,
            num_labels=9,
            id2label=NLPConfig.ID2LABEL,
            label2id={v: k for k, v in NLPConfig.ID2LABEL.items()}
        )
        
        model.to(NLPConfig.DEVICE)
        model.eval()
        
        print("✅ NLP model loaded successfully!")
        print(f"   Device: {NLPConfig.DEVICE}")
        print(f"   Num labels: {model.config.num_labels}")
        print(f"   Model type: {type(model).__name__}")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ ERROR loading NLP model: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        return None, None


def predict_emotion(text, model, tokenizer):
    if not text or not text.strip():
        return None, 0.0

    cleaned_text = light_clean(text)
    print(f"[NLP] Original text: '{text}'")
    print(f"[NLP] Cleaned text: '{cleaned_text}'")
    
    enc = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=NLPConfig.MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = enc['input_ids'].to(NLPConfig.DEVICE)
    attention_mask = enc['attention_mask'].to(NLPConfig.DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get logits
        logits = outputs.logits
        
        # Apply temperature scaling for better confidence calibration
        scaled_logits = logits / NLPConfig.TEMPERATURE
        
        # Get probabilities using softmax on scaled logits
        probs = torch.nn.functional.softmax(scaled_logits, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(logits, dim=1).cpu().item()  # Still use original logits for prediction
        confidence = probs[0][pred_idx].cpu().item()
        
        # Debug: print top 3 predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(3, probs.shape[1]))
        print(f"[NLP DEBUG] Top 3 predictions (with temperature={NLPConfig.TEMPERATURE}):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            emotion_name = NLPConfig.ID2LABEL.get(idx.item(), "Unknown")
            print(f"  {i+1}. {emotion_name}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    emotion = NLPConfig.ID2LABEL.get(pred_idx, "Unknown")
    return emotion, confidence