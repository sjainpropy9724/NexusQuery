import sqlite3
import json

DB_PATH = "chat_history.db"

def initialize_db():
    """Creates the chat_history table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kb_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


def add_message_to_history(kb_id: str, role: str, content: str):
    """Adds a new message to the chat history for a given KB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (kb_id, role, content) VALUES (?, ?, ?)",
        (kb_id, role, content)
    )
    conn.commit()
    conn.close()

def get_history(kb_id: str) -> list[dict]:
    """Retrieves the chat history for a given KB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE kb_id = ? ORDER BY id ASC",
        (kb_id,)
    )
    history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return history