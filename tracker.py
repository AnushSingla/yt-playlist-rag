import os
import sqlite3
from typing import Optional

DB_PATH = os.getenv("DB_PATH", "data/pipeline_state.db")

class PipelineTracker:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    duration INTEGER,
                    status TEXT DEFAULT 'PENDING',
                    source TEXT
                )
            """)

    def add_video(self, video_id: str, title: str, duration: int):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, title, duration) VALUES (?, ?, ?)",
                (video_id, title, duration)
            )

    def update_status(self, video_id: str, status: str, source: Optional[str] = None):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE videos SET status = ?, source = COALESCE(?, source) WHERE video_id = ?",
                (status, source, video_id)
            )

    def get_pending(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT video_id, title FROM videos WHERE status = 'PENDING'")
            return cursor.fetchall()