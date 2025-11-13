"""
Create Catto-Lingo tables in the database
"""
from database import Base, engine
import models

print("🚀 Creating Catto-Lingo tables...")

try:
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")
    print("📋 Created tables:")
    print("  - catto_users")
    print("  - catto_cats")
    print("  - catto_predictions")
except Exception as e:
    print(f"❌ Error: {e}")
