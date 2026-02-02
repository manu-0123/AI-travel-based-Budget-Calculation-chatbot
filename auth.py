# auth.py
# Secure authentication logic

import hashlib
import sqlite3
from db import get_connection

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, email, password):
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, email, hash_password(password))
        )
        conn.commit()
        return True, "Account created successfully"
    except sqlite3.IntegrityError:
        return False, "Username or Email already exists"
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def login(username, password):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT user_id, password_hash
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    row = cur.fetchone()
    conn.close()

    if not row:
        return False, None

    user_id, stored_hash = row

    if stored_hash == hash_password(password):
        return True, user_id

    return False, None 