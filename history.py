# history.py
# History Management for storing and retrieving user search history.
import sqlite3
import pandas as pd
from db import get_connection

def save_search(user_id, data):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO search_history (
            user_id, source, destination, travel_mode,
            duration_days, num_people, temperature,
            rain_flag, total_cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        data["Source"],
        data["Destination"],
        data["Travel_Mode"],
        data["Duration_Days"],
        data["Num_People"],
        data["Temperature"],
        data["Rain_Flag"],
        data["Total_Cost"]
    ))

    conn.commit()
    conn.close()


def load_user_history(user_id):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM search_history WHERE user_id = ?",
        conn,
        params=(user_id,)
    )
    conn.close()
    return df


def delete_history_item(history_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM search_history WHERE history_id = ?", (history_id,))
    conn.commit()
    conn.close()


def clear_user_history(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()