from sqlalchemy import Column, Integer, String, Float, ForeignKey, TIMESTAMP, Text, VARCHAR
from sqlalchemy.sql.expression import text
from sqlalchemy.orm import relationship
from database import Base
class User(Base):
    __tablename__ = "catto_users"  # اسم مختلف عن users table بتاع Supabase

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(VARCHAR(50), nullable=False, unique=True)
    email = Column(VARCHAR(100), unique=True)
    password_hash = Column(VARCHAR(255), nullable=True)  # لتخزين الـ hashed password (nullable للتوافق مع البيانات القديمة)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    cats = relationship("Cat", back_populates="owner")
    predictions = relationship("Prediction", back_populates="user")
class Cat(Base):
    __tablename__ = "catto_cats"

    cat_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("catto_users.user_id", ondelete="CASCADE"), nullable=False)
    cat_name = Column(VARCHAR(100), nullable=False)
    breed = Column(VARCHAR(100))
    age = Column(Integer)
    notes = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))

    
    owner = relationship("User", back_populates="cats")
    predictions = relationship("Prediction", back_populates="cat")


class Prediction(Base):
    __tablename__ = "catto_predictions"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("catto_users.user_id"), nullable=False)
    cat_id = Column(Integer, ForeignKey("catto_cats.cat_id"), nullable=True) # ممكن يبقى اختياري
    prediction_time = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'))
    input_type = Column(VARCHAR(10), nullable=False)
    detected_emotion = Column(VARCHAR(50))
    model_confidence = Column(Float)
    user_feedback = Column(VARCHAR(50))
    agent_response = Column(Text)
    model_used = Column(VARCHAR(50))

  
    user = relationship("User", back_populates="predictions")
    cat = relationship("Cat", back_populates="predictions")