# استخدام Python 3.10 كـ base image
FROM python:3.10-slim

# تحديد مجلد العمل داخل الـ container
WORKDIR /app

# تثبيت المكتبات النظامية المطلوبة
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# نسخ ملف المتطلبات أولاً للاستفادة من Docker cache
COPY requirements.txt .

# تثبيت المكتبات Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# نسخ كل ملفات المشروع
COPY . .

# إنشاء مجلدات للـ models و cache
RUN mkdir -p /app/models /app/cache /app/uploads

# تحديد متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV HUGGINGFACE_HUB_CACHE=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# فتح البورت 8001
EXPOSE 8001

# تشغيل السيرفر
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
