import sqlite3
import json

DB_PATH = "faces.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def save_face(name: str, embedding: list):
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)",
                (name, json.dumps(embedding)))
    con.commit()
    con.close()

def get_all_faces():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, name, embedding FROM faces").fetchall()
    con.close()
    return rows

def delete_face_by_id(face_id: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM faces WHERE id = ?", (face_id,))
    con.commit()
    con.close()
