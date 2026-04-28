#!/usr/bin/env python3
"""
Alembic Migration Diagnostic Script
Run this INSIDE the Docker container to check migration status
"""

import sys
import os

# Ensure app imports work
sys.path.insert(0, '/app')

print("=" * 70)
print("🔍 ALEMBIC MIGRATION DIAGNOSTIC TOOL")
print("=" * 70)

# ── Step 1: Load Models ────────────────────────────────────────
print("\n📦 STEP 1: Loading SQLAlchemy Models...")
print("-" * 70)

try:
    from app.models.domain import Base
    
    model_tables = sorted(Base.metadata.tables.keys())
    print(f"✅ Models loaded successfully!")
    print(f"   Total tables defined: {len(model_tables)}\n")
    
    for table_name in model_tables:
        print(f"      ➤ {table_name}")
        
except Exception as e:
    print(f"❌ ERROR loading models!")
    print(f"   Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Step 2: Test Database Connection ──────────────────────────
print("\n" + "=" * 70)
print("📦 STEP 2: Testing Database Connection...")
print("-" * 70)

try:
    from app.core.config import get_settings
    from sqlalchemy import create_engine, text, inspect
    
    settings = get_settings()
    db_url = settings.DATABASE_URL_SYNC
    
    # Hide password in output
    safe_url = db_url.split('@')[0] + '@***' if '@' in db_url else db_url
    print(f"✅ Configuration loaded")
    print(f"   Database URL: {safe_url}")
    
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Get DB info
        result = conn.execute(text("SELECT current_database(), version(), current_user"))
        db_name, pg_version, db_user = result.fetchone()
        
        print(f"\n✅ Connected to database successfully!")
        print(f"   Database: {db_name}")
        print(f"   User: {db_user}")
        print(f"   PostgreSQL: {pg_version[:60]}...")
        
        # Get existing tables
        inspector = inspect(engine)
        existing_tables = sorted(inspector.get_table_names())
        
        print(f"\n   Tables currently in database: {len(existing_tables)}")
        if existing_tables:
            for t in existing_tables:
                print(f"      ➤ {t}")
        else:
            print("      ⚠️  (empty - no tables found)")
            
except Exception as e:
    print(f"\n❌ ERROR connecting to database!")
    print(f"   Error: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ── Step 3: Compare Models vs Database ─────────────────────────
print("\n" + "=" * 70)
print("📦 STEP 3: Comparing Models vs Database")
print("-" * 70)

try:
    model_set = set(model_tables)
    db_set = set(existing_tables)
    
    # Find differences
    new_tables = sorted(model_set - db_set)  # In models but not in DB
    obsolete_tables = sorted(db_set - model_set)  # In DB but not in models
    common_tables = sorted(model_set & db_set)  # In both
    
    print(f"\n📊 Comparison Results:")
    print(f"   Tables in models only (need migration): {len(new_tables)}")
    if new_tables:
        for t in new_tables:
            print(f"      🆕 {t} ← SHOULD CREATE")
    
    print(f"\n   Tables in database only (obsolete?): {len(obsolete_tables)}")
    if obsolete_tables:
        for t in obsolete_tables:
            print(f"      ⚠️  {t}")
    
    print(f"\n   Tables in both (already synced): {len(common_tables)}")
    if common_tables:
        for t in common_tables:
            print(f"      ✅ {t}")
            
except Exception as e:
    print(f"\n❌ ERROR during comparison: {e}")

# ── Step 4: Test Autogenerate Detection ───────────────────────
print("\n" + "=" * 70)
print("📦 STEP 4: Testing Alembic Autogenerate Detection")
print("-" * 70)

try:
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    
    conn = engine.connect()
    context = MigrationContext.configure(conn)
    diff = compare_metadata(context, Base.metadata)
    conn.close()
    
    print(f"\n🔍 Autogenerate detected {len(diff)} changes:")
    
    if diff:
        change_types = {}
        for item in diff:
            change_type = item[0]
            change_types[change_type] = change_types.get(change_type, 0) + 1
            
        for ctype, count in change_types.items():
            print(f"      • {ctype}: {count} occurrence(s)")
            
        print(f"\n   First few changes:")
        for item in diff[:5]:
            print(f"      - {item}")
            
    else:
        print("      ⚠️  NO CHANGES DETECTED")
        print("      → Models match database exactly")
        print("      → This explains why no migration was generated!")
        
except Exception as e:
    print(f"\n❌ ERROR testing autogenerate: {e}")
    import traceback
    traceback.print_exc()

# ── Final Recommendation ──────────────────────────────────────
print("\n" + "=" * 70)
print("🎯 DIAGNOSIS COMPLETE - RECOMMENDATIONS")
print("=" * 70)

if new_tables:
    print("""
┌─────────────────────────────────────────────────────────────┐
│ STATUS: 🆕 NEW TABLES DETECTED                              │
│                                                             │
│ Your models have tables that don't exist in the database.   │
│                                                             │
│ ▶ RUN THIS COMMAND:                                        │
│   alembic revision --autogenerate -m "initial_schema"      │
│                                                             │
│ Or if autogenerate fails, try manual approach:              │
│   alembic revision -m "initial_schema"                     │
└─────────────────────────────────────────────────────────────┘""")
    
elif not existing_tables and model_tables:
    print("""
┌─────────────────────────────────────────────────────────────┐
│ STATUS: 📭 EMPTY DATABASE                                  │
│                                                             │
│ Database exists but has NO tables.                          │
│ All your models need to be created.                         │
│                                                             │
│ ▶ RUN THIS COMMAND:                                        │
│   alembic revision --autogenerate -m "initial_schema"      │
│   alembic upgrade head                                     │
└─────────────────────────────────────────────────────────────┘""")

else:
    print("""
┌─────────────────────────────────────────────────────────────┐
│ STATUS: ✅ DATABASE UP TO DATE                              │
│                                                             │
│ All tables already exist in the database.                   │
│ No migration file was needed because nothing changed!       │
│                                                             │
│ ▶ RUN THIS COMMAND to mark baseline:                       │
│   alembic stamp head                                       │
│                                                             │
│ Future schema changes will now generate migrations.         │
└─────────────────────────────────────────────────────────────┘""")

print("\n" + "=" * 70)
print("✅ Diagnostic complete!")
print("=" * 70 + "\n")