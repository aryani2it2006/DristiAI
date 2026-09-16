"""
engine/alert_engine.py

Implements Module I (Section 11) / Alert Engine (Section 65).

Section 11.2 workflow: Event -> Rule Check -> Alert-worthy? -> Generate
Alert. The spec does not enumerate the exact rule set, so the rules
below are a reasonable, explicit reading of Section 2.3 / 3.2's own list
of "important" event types -- flagged so you can adjust which event
types should page a human operator.
"""

ALERT_WORTHY_TYPES = {
    "INTRUSION",
    "SUSPICIOUS_ACTIVITY",
    "NIGHT_MOVEMENT",
}


def evaluate_event(event):
    """
    Returns an alert dict if the event should raise an alert, else None.
    """
    if event["event_type"] not in ALERT_WORTHY_TYPES:
        return None

    return {
        "alert_id": f"ALT-{event['event_id'].split('-')[1]}",
        "event_id": event["event_id"],
        "event_type": event["event_type"],
        "camera_id": event["camera_id"],
        "confidence": event["confidence"],
        "timestamp": event["timestamp"],
        "message": f"{event['event_type']} detected on {event['camera_id']}",
    }
