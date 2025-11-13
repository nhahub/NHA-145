from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

# ============= Authentication Schemas =============
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[int] = None

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

# ============= User Schemas =============
class UserBase(BaseModel):
    username: str
    email: Optional[EmailStr] = None


class UserCreate(UserBase):
    password: str


class User(UserBase):
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True 


class CatBase(BaseModel):
    cat_name: str
    breed: Optional[str] = None
    age: Optional[int] = None
    notes: Optional[str] = None

class CatCreate(CatBase):
    pass

class Cat(CatBase):
    cat_id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True



class PredictionBase(BaseModel):
    input_type: str
    detected_emotion: str
    model_confidence: float
    agent_response: Optional[str] = None
    model_used: Optional[str] = None

class PredictionCreate(PredictionBase):
    user_id: int 
    cat_id: Optional[int] = None 

class Prediction(PredictionBase):
    prediction_id: int
    user_id: int
    cat_id: Optional[int] = None
    prediction_time: datetime
    user_feedback: Optional[str] = None

    class Config:
        from_attributes = True