-- MindGuard v2.0 Database Schema
-- All tables linked via anonymized user_id (UUID)

-- Users table (anonymized)
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_text TEXT NOT NULL,
    risk_level TEXT CHECK(risk_level IN ('Low', 'Medium', 'High')) NOT NULL,
    confidence REAL CHECK(confidence >= 0.0 AND confidence <= 1.0) NOT NULL,
    shap_summary JSON,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Clinical flags table
CREATE TABLE IF NOT EXISTS clinical_flags (
    flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    indicator_type TEXT NOT NULL,
    matched_keywords TEXT,
    severity TEXT CHECK(severity IN ('Low', 'Medium', 'High')) NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Trends table (daily aggregation)
CREATE TABLE IF NOT EXISTS trends (
    trend_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    date DATE NOT NULL,
    avg_risk_score REAL CHECK(avg_risk_score >= 0.0 AND avg_risk_score <= 3.0),
    dominant_flag TEXT,
    session_count INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE(user_id, date)
);

-- LSTM predictions table
CREATE TABLE IF NOT EXISTS lstm_predictions (
    pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    forecast_json JSON NOT NULL,
    confidence REAL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp);
CREATE INDEX IF NOT EXISTS idx_clinical_flags_session_id ON clinical_flags(session_id);
CREATE INDEX IF NOT EXISTS idx_trends_user_id ON trends(user_id);
CREATE INDEX IF NOT EXISTS idx_trends_date ON trends(date);
CREATE INDEX IF NOT EXISTS idx_trends_user_date ON trends(user_id, date);
CREATE INDEX IF NOT EXISTS idx_lstm_predictions_user_id ON lstm_predictions(user_id);