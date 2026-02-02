# database (sql lite3) setup for backend fundamentals(User Account Accessories )
# This file creates tables automatically.
# SQLite database setup for User Accounts & Search History

import sqlite3
import os

DB_PATH = "users.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # USERS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # SEARCH HISTORY TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            destination TEXT,
            travel_mode TEXT,
            duration_days INTEGER,
            num_people INTEGER,
            temperature REAL,
            rain_flag INTEGER,
            total_cost REAL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()

def migrate_users_table():
    conn = get_connection()
    cur = conn.cursor()

    # Check existing columns
    cur.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cur.fetchall()]

    # Add password_hash if missing
    if "password_hash" not in columns:
        cur.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        conn.commit()

    conn.close()
