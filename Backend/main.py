import sys
import os
import tempfile
import shutil
import io # Needed for image endpoint
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Dict, Any, List
from PIL import Image # Needed for image endpoint
import numpy as np # Needed for image endpoint
import cv2 # Needed for image endpoint
import pandas as pd # Needed for video summary
import librosa # quick audio validation

from sqlalchemy.orm import Session
import models
import schemas
import database
import agent # ده الـ Agent بتاع Gemini
import auth # ده ملف Authentication الجديد
from database import get_db, engine # هنجيب الـ session والـ engine

# --- (جديد) ده بيخلي SQLAlchemy تعمل الجداول لو مش موجودة أول ما نشغل ---
models.Base.metadata.create_all(bind=engine)

os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:/huggingface_cache'
# --------------------------------------------------------------------

# الملفات موجودة في نفس الفولدر backend/
from cattolingo_nlp2_src import load_nlp_model, predict_emotion
# Audio import سيكون lazy (عند الطلب فقط) لتجنب مشاكل TensorFlow/protobuf
# from cattolingo_Audio_cnn_src import load_cnn_model, get_audio_prediction
from cattolingo_Emotion_Detector_src import load_classification_model, process_video_frames, predict_frame_emotion 

# --- إنشاء تطبيق FastAPI ---
app = FastAPI(title="Catto-Lingo Multi-Modal API")

# --- إضافة CORS Middleware ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج: حدد الـ origins المسموح بها
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- متغيرات لتخزين الموديلات المحملة ---
nlp_model, nlp_tokenizer = None, None
audio_cnn_model = None
video_resnet_model = None

# --- تحميل كل الموديلات مرة واحدة عند التشغيل ---
@app.on_event("startup")
async def startup_event():
    global nlp_model, nlp_tokenizer, audio_cnn_model, video_resnet_model
    print(" API Starting Up - Loading Models...")

    # تحميل موديل NLP
    try:
        loaded_items = load_nlp_model()
        if isinstance(loaded_items, tuple) and len(loaded_items) >= 2:
            nlp_model, nlp_tokenizer = loaded_items[0], loaded_items[1]
            if nlp_model is None or nlp_tokenizer is None: raise RuntimeError("NLP model/tokenizer missing")
            print("✅ NLP Model loaded.")
        else: raise RuntimeError("load_nlp_model unexpected return")
    except Exception as e:
        print(f"🔥 ERROR loading NLP model: {e}")

    # تحميل موديل الصوت CNN (lazy load)
    try:
        from cattolingo_Audio_cnn_src import load_cnn_model
        audio_cnn_model = load_cnn_model()
        if audio_cnn_model is None: raise RuntimeError("Audio CNN model missing")
        print("✅ Audio CNN Model loaded.")
    except Exception as e:
        audio_cnn_model = None
        print(f"⚠️ Audio CNN model not loaded: {e}")

    # تحميل موديل الفيديو ResNet
    try:
        model, success, error = load_classification_model()
        if not success or model is None: raise RuntimeError(f"Video ResNet model missing: {error}")
        video_resnet_model = model
        print("✅ Video ResNet Model loaded.")
    except Exception as e:
        print(f"🔥 ERROR loading Video ResNet model: {e}")

    # --- (جديد) تحميل قاعدة المعرفة بتاعة الـ Agent ---
    print("🧠 Loading Knowledge Base...")
    agent.load_knowledge_base() 
    # (ملف agent.py هو اللي هيحمّل مفتاح Gemini)

    print("🚀 Model Loading Complete.")

# --- (تعديل) تعريف شكل الطلبات ---
class TextRequest(BaseModel):
    text: str
    user_id: int # (تعديل) بقى رقم صحيح إجباري

# --- Endpoint الترحيب ---
@app.get("/")
def read_root():
    return {"message": "أهلاً بك في Catto-Lingo Multi-Modal API!"}


# ============= Authentication Endpoints =============

@app.post("/auth/register", response_model=schemas.User, summary="Register a new user")
def register(user: schemas.UserRegister, db: Session = Depends(get_db)):
    """تسجيل مستخدم جديد مع تشفير كلمة المرور"""
    # التحقق من عدم وجود اسم المستخدم
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="اسم المستخدم موجود بالفعل")
    
    # التحقق من عدم وجود البريد الإلكتروني
    db_email = db.query(models.User).filter(models.User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="البريد الإلكتروني مسجل بالفعل")
    
    # إنشاء المستخدم الجديد
    new_user = models.User(
        username=user.username,
        email=user.email,
        password_hash=auth.get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/auth/login", response_model=schemas.Token, summary="Login with username and password")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """تسجيل الدخول والحصول على JWT token"""
    # البحث عن المستخدم
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="اسم المستخدم أو كلمة المرور غير صحيحة"
        )
    
    # إنشاء JWT token - convert user_id to string for JWT standard
    access_token = auth.create_access_token(data={"sub": str(user.user_id)})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/auth/me", response_model=schemas.User, summary="Get current user info")
def get_current_user_info(current_user: models.User = Depends(auth.get_current_user)):
    """الحصول على معلومات المستخدم الحالي"""
    return current_user


@app.get("/auth/test-token", summary="Test token parsing (DEBUG)")
async def test_token(authorization: str = Header(None), db: Session = Depends(get_db)):
    """Test endpoint to debug token issues"""
    if not authorization:
        return {"error": "No Authorization header"}
    
    if not authorization.startswith("Bearer "):
        return {"error": "Invalid Authorization header format", "received": authorization}
    
    token = authorization.replace("Bearer ", "")
    
    try:
        from jose import jwt
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        user_id = payload.get("sub")
        
        user = db.query(models.User).filter(models.User.user_id == user_id).first()
        
        return {
            "token": token[:50] + "...",
            "payload": payload,
            "user_id": user_id,
            "user_found": user is not None,
            "user": {"username": user.username, "user_id": user.user_id} if user else None
        }
    except Exception as e:
        return {"error": str(e)}


# ============= User Management Endpoints =============

@app.post("/users/", response_model=schemas.User, summary="Create a new user (deprecated - use /auth/register)")
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """⚠️ Deprecated: استخدم /auth/register بدلاً من ذلك"""
    # شوف اليوزر موجود قبل كده ولا لأ
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    # اعمل اليوزر الجديد
    new_user = models.User(
        username=user.username,
        email=user.email,
        password_hash=auth.get_password_hash(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/users/{user_id}", response_model=schemas.User, summary="Get user details")
def read_user(user_id: int, db: Session = Depends(get_db)): 
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# --- (تعديل كامل) Endpoint التنبؤ بالنص ---
@app.post("/predict/text", summary="Predict emotion from text and get AI advice")
async def predict_text_endpoint(request: TextRequest, db: Session = Depends(get_db)):
    if nlp_model is None or nlp_tokenizer is None:
        raise HTTPException(status_code=503, detail="NLP Model is not available.")
    
    print(f"Received text request for user {request.user_id}: '{request.text}'")
    
    # نتأكد إن اليوزر موجود في الداتابيز
    db_user = db.query(models.User).filter(models.User.user_id == request.user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
        
    try:
        # --- 1. التنبؤ ---
        result = predict_emotion(request.text, nlp_model, nlp_tokenizer)
        if result is None or result[0] is None:
            raise ValueError("Prediction failed")
        
        emotion, confidence = result
        print(f"--> NLP Prediction: {emotion} (confidence: {confidence:.2%})")

        # --- 2. استدعاء الـ Agent (اللي بيكلم Gemini) ---
        agent_response = agent.get_simple_advice(emotion, request.user_id)
        
        # --- 3. تسجيل النتيجة في الداتابيز ---
        new_prediction = models.Prediction(
            user_id=request.user_id,
            input_type='text',
            detected_emotion=emotion,
            model_confidence=confidence,
            agent_response=agent_response,
            model_used='roberta_nlp' # (أو أي اسم تحبه)
        )
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)

        # --- 4. إرجاع الرد ---
        return {
            "emotion": emotion,  # Changed from detected_emotion to match frontend
            "confidence": confidence,  # Added confidence
            "agent_response": agent_response,
            "prediction_id": new_prediction.prediction_id
        }
        
    except Exception as e:
        print(f"🔥 NLP Prediction Error: {e}")
        db.rollback() # (جديد) لو حصل خطأ، نتراجع عن أي حاجة في الداتابيز
        raise HTTPException(status_code=500, detail=str(e))

# --- (تعديل كامل) Endpoint التنبؤ بالصوت ---
@app.post("/predict/audio", summary="Predict emotion from audio and get AI advice")
async def predict_audio_endpoint(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if audio_cnn_model is None:
        raise HTTPException(status_code=503, detail="Audio CNN Model is not available.")

    print(f"Received audio request for user {user_id}: '{file.filename}' ({file.content_type})")
    
    # نتأكد إن اليوزر موجود
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename or "temp_audio")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Audio file saved temporarily to: {temp_path}")

        # Basic validation
        try:
            _dur = librosa.get_duration(path=temp_path)
            if _dur is None or _dur == 0: raise ValueError("Audio duration is zero.")
        except Exception as ve:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {ve}")

        # --- 1. التنبؤ ---
        from cattolingo_Audio_cnn_src import get_audio_prediction
        emotion, confidence = get_audio_prediction(audio_cnn_model, temp_path)
        if emotion is None: raise ValueError("Prediction failed")
        print(f"--> Audio Prediction: {emotion} (Confidence: {confidence:.2f})")

        # --- 2. استدعاء الـ Agent (اللي بيكلم Gemini) ---
        agent_response = agent.get_simple_advice(emotion, user_id)

        # --- 3. تسجيل النتيجة في الداتابيز ---
        new_prediction = models.Prediction(
            user_id=user_id,
            input_type='audio',
            detected_emotion=emotion,
            model_confidence=float(confidence),
            agent_response=agent_response,
            model_used='audio_cnn'
        )
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)

        # --- 4. إرجاع الرد ---
        return {
            "emotion": emotion,  # Changed from detected_emotion
            "confidence": float(confidence),
            "agent_response": agent_response,
            "prediction_id": new_prediction.prediction_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Audio Prediction Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        await file.close()

# --- (تعديل كامل) Endpoint التنبؤ بالفيديو ---
@app.post("/predict/video", summary="Predict dominant emotion from video and get AI advice")
async def predict_video_endpoint(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if video_resnet_model is None:
        raise HTTPException(status_code=503, detail="Video ResNet Model is not available.")

    print(f"Received video request for user {user_id}: '{file.filename}' ({file.content_type})")
    
    # نتأكد إن اليوزر موجود
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename or "temp_video")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Video file saved temporarily to: {temp_path}")

        # Quick sanity check
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            cap.release()
            raise HTTPException(status_code=400, detail="Invalid or unsupported video file.")
        cap.release()

        # --- 1. التنبؤ ---
        # (ده بينادي الملف اللي على اليمين 'cattolingo_Emotion_Detector_src.py')
        output_video_path, emotions_summary_counter = process_video_frames(temp_path, video_resnet_model)
        
        if not emotions_summary_counter:
            # (ده غالباً اللي كان بيحصل، بيرجع 'N/A' لو الملف باظ)
            print("🔥 Video processing returned empty summary. Check 'cattolingo_Emotion_Detector_src.py'.")
            raise ValueError("Video processing failed or no emotions detected.")

        df_summary = pd.DataFrame(emotions_summary_counter.items(), columns=['Emotion', 'Frame Count'])
        
        df_summary = df_summary[df_summary['Emotion'] != 'N/A']
        
        if df_summary.empty:
            print("All frames returned N/A - model predictions failed for entire video.")
            raise ValueError("Video analysis failed: all frames returned N/A. Check video quality or model.")
        
        total_frames = int(df_summary['Frame Count'].sum())
        if total_frames > 0:
            df_summary['Percentage'] = (df_summary['Frame Count'] / total_frames) * 100
        else:
            df_summary['Percentage'] = 0.0

        # Get the dominant emotion (most frequent)
        dominant = df_summary.loc[df_summary['Frame Count'].idxmax()]
        dominant_emotion = str(dominant['Emotion'])
        confidence = float(dominant.get('Percentage', 0.0)) / 100.0 # (بنجيب نسبة الثقة من نسبة الفريمات)

        summary_dict = df_summary.set_index('Emotion').to_dict('index')
        print(f"--> Video Analysis Summary: Dominant={dominant_emotion}, Details={summary_dict}")

        # --- 2. استدعاء الـ Agent (اللي بيكلم Gemini) ---
        agent_response = agent.get_simple_advice(dominant_emotion, user_id)

        # --- 3. تسجيل النتيجة في الداتابيز ---
        new_prediction = models.Prediction(
            user_id=user_id,
            input_type='video',
            detected_emotion=dominant_emotion,
            model_confidence=confidence,
            agent_response=agent_response,
            model_used='video_resnet'
        )
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)

        # --- 4. إرجاع الرد ---
        return {
            "emotion": dominant_emotion,
            "confidence": float(confidence),  # Use the calculated confidence from Percentage
            "agent_response": agent_response,
            "emotion_summary": summary_dict,
            "prediction_id": new_prediction.prediction_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Video Prediction Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        await file.close()

# --- ( ( ( ( ( التعديل اللي إنت عايزه هنا ) ) ) ) ) ---
@app.post("/predict/image", summary="Predict emotion from image and get AI advice")
async def predict_image_endpoint(user_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if video_resnet_model is None:
        raise HTTPException(status_code=503, detail="Image/Video ResNet Model is not available.")

    print(f"Received image request for user {user_id}: '{file.filename}' ({file.content_type})")

    # نتأكد إن اليوزر موجود
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not file.content_type or not str(file.content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type, expected image/* content-type.")

    try:
        # --- ( ( ( الكود الجديد النضيف ) ) ) ---
        # 1. اقرا الصورة (بايتس)
        image_bytes = await file.read()
        
        # 2. حول البايتس لـ numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # 3. حول الـ numpy array لـ frame (بتاع cv2)
        # ده الأمر الصح 100%
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image file. Is it corrupted?")
        # --- ( ( ( نهاية الكود الجديد ) ) ) ---

        # --- 1. التنبؤ ---
        # (ده بينادي الملف اللي على اليمين 'cattolingo_Emotion_Detector_src.py')
        # وهو مستني (frame_bgr) بالظبط
        emotion, confidence = predict_frame_emotion(video_resnet_model, frame_bgr)
        
        if emotion is None or emotion == "N/A": 
            print(f"🔥 Prediction returned N/A. Check model logic or input image.")
            raise ValueError("Prediction failed, model returned N/A.")
            
        print(f"--> Image Prediction: {emotion} (Confidence: {confidence:.2f})")

        # --- 2. استدعاء الـ Agent (اللي بيكلم Gemini) ---
        agent_response = agent.get_simple_advice(emotion, user_id)

        # --- 3. تسجيل النتيجة في الداتابيز ---
        new_prediction = models.Prediction(
            user_id=user_id,
            input_type='image',
            detected_emotion=emotion,
            model_confidence=float(confidence),
            agent_response=agent_response,
            model_used='video_resnet_image'
        )
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)
        
     
        return {
            "emotion": emotion,  # Changed from detected_emotion
            "confidence": float(confidence),
            "agent_response": agent_response,
            "prediction_id": new_prediction.prediction_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f" Image Prediction Error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


