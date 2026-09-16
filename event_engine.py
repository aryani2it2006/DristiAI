"""
engine/event_engine.py

Implements the Event Object (Section 63) and Event Logging (Module J,
Section 12). An Event is the structured record produced whenever an AI
module reports something worth recording.
"""

import itertools
import json
from datetime import datetime, timezone

_id_counter = itertools.count(1)


def make_event(event_type, camera_id, confidence, metadata=None):
    """
    Fields mirror the example event JSON in Section 34.1:
        event_id, event_type, camera_id, confidence, timestamp
    plus a metadata dict for module-specific details (Section 12.2).
    """
    event_id = f"EVT-{next(_id_counter):04d}"
    return {
        "event_id": event_id,
        "event_type": event_type,
        "camera_id": camera_id,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }


def event_to_json(event):
    return json.dumps(event)
