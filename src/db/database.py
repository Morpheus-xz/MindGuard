"""SQLite database operations for MindGuard."""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mindguard")


class Database:
    """SQLite database handler for MindGuard."""

    def __init__(self, db_path: str = "data/mindguard.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def init_schema(self, schema_path: str = "src/db/schema.sql") -> None:
        """
        Initialize database schema from SQL file.

        Args:
            schema_path: Path to schema SQL file.
        """
        schema_file = Path(schema_path)

        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_file, "r") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            logger.info("Database schema initialized successfully")

    # ==================== USER OPERATIONS ====================

    def create_user(self, user_id: str) -> str:
        """
        Create a new anonymous user.

        Args:
            user_id: Unique anonymous user ID (UUID).

        Returns:
            The created user_id.
        """
        with self.get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
                (user_id,)
            )
        return user_id

    def update_user_activity(self, user_id: str) -> None:
        """
        Update user's last_active timestamp.

        Args:
            user_id: User ID to update.
        """
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,)
            )

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID.

        Args:
            user_id: User ID to fetch.

        Returns:
            User dict or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # ==================== SESSION OPERATIONS ====================

    def create_session(
            self,
            user_id: str,
            input_text: str,
            risk_level: str,
            confidence: float,
            shap_summary: Optional[Dict] = None
    ) -> int:
        """
        Create a new screening session.

        Args:
            user_id: User ID.
            input_text: The text that was analyzed.
            risk_level: Low, Medium, or High.
            confidence: Model confidence (0.0 - 1.0).
            shap_summary: SHAP explanation data.

        Returns:
            The created session_id.
        """
        # Ensure user exists
        self.create_user(user_id)
        self.update_user_activity(user_id)

        shap_json = json.dumps(shap_summary) if shap_summary else None

        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (user_id, input_text, risk_level, confidence, shap_summary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, input_text, risk_level, confidence, shap_json)
            )
            return cursor.lastrowid

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get session by ID.

        Args:
            session_id: Session ID to fetch.

        Returns:
            Session dict or None if not found.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get("shap_summary"):
                    result["shap_summary"] = json.loads(result["shap_summary"])
                return result
            return None

    def get_user_sessions(
            self,
            user_id: str,
            limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent sessions for a user.

        Args:
            user_id: User ID.
            limit: Maximum number of sessions to return.

        Returns:
            List of session dicts.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT *
                FROM sessions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                    LIMIT ?
                """,
                (user_id, limit)
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                result = dict(row)
                if result.get("shap_summary"):
                    result["shap_summary"] = json.loads(result["shap_summary"])
                results.append(result)
            return results

    # ==================== CLINICAL FLAGS OPERATIONS ====================

    def add_clinical_flag(
            self,
            session_id: int,
            indicator_type: str,
            matched_keywords: str,
            severity: str
    ) -> int:
        """
        Add a clinical flag to a session.

        Args:
            session_id: Session ID.
            indicator_type: Type of indicator (e.g., 'hopelessness', 'isolation').
            matched_keywords: Keywords that triggered the flag.
            severity: Low, Medium, or High.

        Returns:
            The created flag_id.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO clinical_flags (session_id, indicator_type, matched_keywords, severity)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, indicator_type, matched_keywords, severity)
            )
            return cursor.lastrowid

    def get_session_flags(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Get all clinical flags for a session.

        Args:
            session_id: Session ID.

        Returns:
            List of flag dicts.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM clinical_flags WHERE session_id = ?",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==================== TRENDS OPERATIONS ====================

    def update_trend(
            self,
            user_id: str,
            trend_date: date,
            avg_risk_score: float,
            dominant_flag: Optional[str],
            session_count: int
    ) -> int:
        """
        Update or insert daily trend data.

        Args:
            user_id: User ID.
            trend_date: Date for the trend.
            avg_risk_score: Average risk score for the day.
            dominant_flag: Most frequent clinical indicator.
            session_count: Number of sessions that day.

        Returns:
            The trend_id.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trends (user_id, date, avg_risk_score, dominant_flag, session_count)
                VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id, date) DO
                UPDATE SET
                    avg_risk_score = excluded.avg_risk_score,
                    dominant_flag = excluded.dominant_flag,
                    session_count = excluded.session_count
                """,
                (user_id, trend_date.isoformat(), avg_risk_score, dominant_flag, session_count)
            )
            return cursor.lastrowid

    def get_user_trends(
            self,
            user_id: str,
            days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get trend data for a user.

        Args:
            user_id: User ID.
            days: Number of days to look back.

        Returns:
            List of trend dicts ordered by date.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT *
                FROM trends
                WHERE user_id = ?
                ORDER BY date DESC
                    LIMIT ?
                """,
                (user_id, days)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==================== LSTM PREDICTIONS OPERATIONS ====================

    def save_lstm_prediction(
            self,
            user_id: str,
            forecast_json: List[float],
            confidence: float
    ) -> int:
        """
        Save an LSTM trend prediction.

        Args:
            user_id: User ID.
            forecast_json: 7-day forecast array.
            confidence: Model confidence.

        Returns:
            The prediction_id.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO lstm_predictions (user_id, forecast_json, confidence)
                VALUES (?, ?, ?)
                """,
                (user_id, json.dumps(forecast_json), confidence)
            )
            return cursor.lastrowid

    def get_latest_prediction(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent LSTM prediction for a user.

        Args:
            user_id: User ID.

        Returns:
            Prediction dict or None.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT *
                FROM lstm_predictions
                WHERE user_id = ?
                ORDER BY generated_at DESC LIMIT 1
                """,
                (user_id,)
            )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result["forecast_json"] = json.loads(result["forecast_json"])
                return result
            return None

    # ==================== DATA DELETION ====================

    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user (GDPR compliance).

        Args:
            user_id: User ID to delete.

        Returns:
            True if user existed and was deleted.
        """
        with self.get_connection() as conn:
            # Check if user exists
            cursor = conn.execute(
                "SELECT user_id FROM users WHERE user_id = ?",
                (user_id,)
            )
            if not cursor.fetchone():
                return False

            # Delete user (cascades to all related tables)
            conn.execute(
                "DELETE FROM users WHERE user_id = ?",
                (user_id,)
            )
            logger.info(f"Deleted all data for user: {user_id}")
            return True

    # ==================== UTILITY METHODS ====================

    def get_session_count(self, user_id: str) -> int:
        """
        Get total session count for a user.

        Args:
            user_id: User ID.

        Returns:
            Number of sessions.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM sessions WHERE user_id = ?",
                (user_id,)
            )
            return cursor.fetchone()["count"]