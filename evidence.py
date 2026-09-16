"""
engine/evidence.py

Implements Section 52 -- SHA-256 Evidence Hashing.

Any saved evidence file (e.g. a snapshot frame for an event) gets a
SHA-256 fingerprint. This hash is what later gets recorded on the
blockchain-style ledger (Section 54), so anyone can later prove the
evidence file was not altered.
"""

import hashlib


def hash_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()
