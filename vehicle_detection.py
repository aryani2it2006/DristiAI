"""
ai/vehicle_detection.py

Implements Module C (Section 5) -- Vehicle Detection and Classification.

SPEC GAP: Section 5 describes the workflow (Frame -> Vehicle Detection ->
Vehicle Region -> Classification -> Result) but names NO detection model
(no YOLO, no Haar cascade, nothing). OpenCV does not ship a vehicle
cascade, and I have no network access to fetch one.

DEMO CHOICE (flagged, not spec-mandated): background subtraction +
contour analysis. Any large moving blob is treated as a detected
"vehicle region", and a simple aspect-ratio/size rule stands in for
"classification" (Car vs Unknown). This is enough to prove the pipeline
shape end-to-end on synthetic video -- it is NOT a real vehicle
classifier and should not be used for anything but this trial run.
Confirm the intended real detector before we build the final version.
"""

import cv2

_bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40, detectShadows=False)

MIN_VEHICLE_AREA = 1500


def detect_vehicles(bgr_frame):
    """
    Returns a list of detections:
        {"object": "vehicle_candidate", "vehicle_type": "car" | "unknown",
         "confidence": float, "x1": int, "y1": int, "x2": int, "y2": int}
    """
    fg_mask = _bg_subtractor.apply(bgr_frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_VEHICLE_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h) if h else 0

        # crude heuristic "classification": wide, low blobs -> "car"
        if 1.3 <= aspect <= 4.0:
            vehicle_type = "car"
            confidence = 0.6
        else:
            vehicle_type = "unknown"
            confidence = 0.3

        detections.append({
            "object": "vehicle_candidate",
            "vehicle_type": vehicle_type,
            "confidence": confidence,
            "x1": int(x), "y1": int(y),
            "x2": int(x + w), "y2": int(y + h),
        })
    return detections
