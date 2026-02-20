import sqlite3
import os

# Database path
DB_PATH = "database/digits.db"

# ---------- Initialize Database ----------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS digits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label INTEGER,
            image BLOB
        )
    """)
    conn.commit()
    conn.close()

# ---------- Save one image ----------
def save_image(label: int, image_bytes: bytes):
    """Saves one image with its digit label into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO digits (label, image) VALUES (?, ?)", (label, image_bytes))
    conn.commit()        # ✅ commit after every insert
    conn.close()

# ---------- Load all images ----------
def load_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT label, image FROM digits")
    data = c.fetchall()
    conn.close()
    return data

# ---------- Count images ----------
def count_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM digits")
    total = c.fetchone()[0]
    conn.close()
    return total
def clear_all_data():
    """Deletes all rows from the digits table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM digits")
    conn.commit()
    conn.close()
