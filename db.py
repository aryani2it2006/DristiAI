"""
backend/db.py

*** DEMO STAND-IN FOR MySQL (Section 35) ***

Section 35 requires MySQL. I have no MySQL server available in the
sandbox I tested this in, and no network to reach one. SQLite is used
here ONLY so the demo can persist and query data with zero external
setup -- it is wire-compatible in spirit (same entities: events, alerts,
persons, vehicles per Section 35.2) but the real build must move this to
MySQL with a proper connection layer (database/connection.py per the
Section 33.2 layout) before anything resembling production.

Tables here mirror the entities called out in Section 35.2/35.3/36 as
closely as SQLite allows.
"""

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ibvap_demo.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        event_type TEXT,
        camera_id TEXT,
        confidence REAL,
        timestamp TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY,
        event_id TEXT,
        event_type TEXT,
        camera_id TEXT,
        confidence REAL,
        timestamp TEXT,
        message TEXT
    );

    CREATE TABLE IF NOT EXISTS persons (
        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        status TEXT,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS vehicles (
        vehicle_id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_type TEXT,
        registration_number TEXT,
        description TEXT,
        status TEXT
    );

    CREATE TABLE IF NOT EXISTS blockchain_records (
        event_id TEXT PRIMARY KEY,
        block_index INTEGER,
        evidence_hash TEXT,
        block_hash TEXT,
        previous_hash TEXT,
        timestamp TEXT
    );
    """)
    conn.commit()
    conn.close()


def insert_event(event):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?)",
        (event["event_id"], event["event_type"], event["camera_id"],
         event["confidence"], event["timestamp"], json.dumps(event["metadata"])),
    )
    conn.commit()
    conn.close()


def insert_alert(alert):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?)",
        (alert["alert_id"], alert["event_id"], alert["event_type"], alert["camera_id"],
         alert["confidence"], alert["timestamp"], alert["message"]),
    )
    conn.commit()
    conn.close()


def insert_blockchain_record(event_id, block):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO blockchain_records VALUES (?, ?, ?, ?, ?, ?)",
        (event_id, block["index"], block["evidence_hash"], block["block_hash"],
         block["previous_hash"], block["timestamp"]),
    )
    conn.commit()
    conn.close()


def get_all_events():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM events ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_event(event_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_alerts():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_blockchain_record(event_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM blockchain_records WHERE event_id = ?", (event_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
