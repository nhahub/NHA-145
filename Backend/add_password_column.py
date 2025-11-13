"""
Script to add password_hash column to users table
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in .env file")
    exit(1)

print(f"🔗 Connecting to database...")

# Create engine
engine = create_engine(DATABASE_URL)

try:
    with engine.connect() as conn:
        # Check if column exists
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='users' AND column_name='password_hash'
        """))
        
        if result.fetchone():
            print(" Column 'password_hash' already exists!")
        else:
            print(" Adding 'password_hash' column...")
            conn.execute(text("""
                ALTER TABLE users 
                ADD COLUMN password_hash VARCHAR(255)
            """))
            conn.commit()
            print("✅ Column 'password_hash' added successfully!")
            
except Exception as e:
    print(f" Error: {e}")
    exit(1)

print(" Database migration completed!")
