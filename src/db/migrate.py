"""Database migration script for MindGuard."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.db.database import Database
from src.utils import setup_logging


def main():
    """Run database migration."""
    logger = setup_logging("INFO")

    print("=" * 50)
    print("MindGuard Database Migration")
    print("=" * 50)

    try:
        # Initialize database
        db = Database("data/mindguard.db")
        db.init_schema("src/db/schema.sql")

        print("✅ Database created: data/mindguard.db")
        print("✅ All tables initialized:")
        print("   - users")
        print("   - sessions")
        print("   - clinical_flags")
        print("   - trends")
        print("   - lstm_predictions")
        print("✅ Indexes created")
        print()
        print("Database is ready!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()