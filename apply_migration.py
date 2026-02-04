#!/usr/bin/env python3
"""
Apply auto-capture migration to database.
Run: python apply_migration.py
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

def apply_migration():
    print("Aplicando migração Auto-Capture...")

    conn = psycopg2.connect(DATABASE_URL)
    c = conn.cursor()

    try:
        # 1. Create auto_capture_events table
        print("1. Criando tabela auto_capture_events...")
        c.execute("""
            CREATE TABLE IF NOT EXISTS auto_capture_events (
                id SERIAL PRIMARY KEY,
                owner_key TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data JSONB NOT NULL,
                captured_at TIMESTAMPTZ DEFAULT NOW(),
                processed BOOLEAN DEFAULT FALSE,
                memory_id INTEGER REFERENCES memories(id)
            )
        """)

        # 2. Create indexes
        print("2. Criando índices...")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auto_capture_session ON auto_capture_events(session_id, captured_at DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auto_capture_processed ON auto_capture_events(processed, captured_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auto_capture_owner ON auto_capture_events(owner_key)")

        # 3. Add column to tier_definitions (check if exists first)
        print("3. Adicionando coluna max_auto_capture_per_day...")
        c.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='tier_definitions' AND column_name='max_auto_capture_per_day'
                ) THEN
                    ALTER TABLE tier_definitions ADD COLUMN max_auto_capture_per_day INTEGER DEFAULT 1000;
                END IF;
            END $$;
        """)

        # 4. Update tier limits
        print("4. Atualizando limites dos tiers...")
        c.execute("UPDATE tier_definitions SET max_auto_capture_per_day = 100 WHERE tier = 'free'")
        c.execute("UPDATE tier_definitions SET max_auto_capture_per_day = 5000 WHERE tier = 'pro'")
        c.execute("UPDATE tier_definitions SET max_auto_capture_per_day = 50000 WHERE tier = 'team'")
        c.execute("UPDATE tier_definitions SET max_auto_capture_per_day = -1 WHERE tier = 'root'")

        conn.commit()
        print("\n✅ Migração aplicada com sucesso!")

        # Verify
        c.execute("SELECT COUNT(*) FROM auto_capture_events")
        print(f"✅ Tabela auto_capture_events: {c.fetchone()[0]} eventos")

        c.execute("SELECT tier, max_auto_capture_per_day FROM tier_definitions ORDER BY priority")
        print("\n✅ Limites de auto-capture por tier:")
        for row in c.fetchall():
            limit_str = "unlimited" if row[1] == -1 else f"{row[1]}/dia"
            print(f"   - {row[0]}: {limit_str}")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Erro ao aplicar migração: {e}")
        raise
    finally:
        c.close()
        conn.close()

if __name__ == "__main__":
    apply_migration()
