"""
Quick test to check database schema and create a test user
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found")
    exit(1)

engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check table columns
        result = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name='users'
            ORDER BY ordinal_position
        """))
        
        print("📋 Users table columns:")
        for row in result:
            print(f"  - {row[0]}: {row[1]} (nullable: {row[2]})")
        
        print("\n" + "="*50)
        
        # Try to insert a test user
        print("\n🧪 Testing user creation...")
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        test_hash = pwd_context.hash("testpassword123")
        
        try:
            conn.execute(text("""
                INSERT INTO users (username, email, password_hash)
                VALUES (:username, :email, :password_hash)
            """), {"username": "test_user_direct", "email": "testdirect@test.com", "password_hash": test_hash})
            conn.commit()
            print("✅ Test user created successfully!")
        except Exception as e:
            print(f"❌ Failed to create user: {e}")
            
except Exception as e:
    print(f"❌ Database error: {e}")
