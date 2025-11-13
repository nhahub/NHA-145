# auth.py - JWT Authentication Utilities

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import models
from database import SessionLocal

# Secret key for JWT (في الإنتاج: استخدم environment variable)
SECRET_KEY = "your-secret-key-here-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storing."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Get the current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"[AUTH DEBUG] Token payload: {payload}")
        user_id_str: str = payload.get("sub")
        print(f"[AUTH DEBUG] Extracted user_id (string): {user_id_str}, type: {type(user_id_str)}")
        if user_id_str is None:
            print("[AUTH DEBUG] user_id is None!")
            raise credentials_exception
        
        # Convert string to int
        try:
            user_id = int(user_id_str)
            print(f"[AUTH DEBUG] Converted user_id (int): {user_id}")
        except (ValueError, TypeError):
            print(f"[AUTH DEBUG] Failed to convert user_id to int: {user_id_str}")
            raise credentials_exception
            
    except JWTError as e:
        print(f"[AUTH DEBUG] JWT Error: {e}")
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    print(f"[AUTH DEBUG] Found user: {user}")
    if user is None:
        print("[AUTH DEBUG] User not found in database!")
        raise credentials_exception
    
    return user


# Optional: For endpoints that work with or without authentication
async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Get current user if authenticated, None otherwise."""
    if not token:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            return None
        
        # Convert string to int
        try:
            user_id = int(user_id_str)
        except (ValueError, TypeError):
            return None
        
        user = db.query(models.User).filter(models.User.user_id == user_id).first()
        return user
    except JWTError:
        return None
