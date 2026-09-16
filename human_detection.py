"""
ai/human_detection.py

Implements Module B (Section 4) -- Human Detection and Tracking.

SPEC GAP: Section 4 describes the input/output of human detection
(bounding box + confidence) but does NOT name a specific detection model
or library. For this demo we use OpenCV's built-in HOG + default people
detector because it ships with OpenCV (already installed, no download
needed) and returns exactly the fields the spec asks for: object name,
confidence, x1, y1, x2, y2.

This is OUR choice to make the demo runnable offline -- flag for
confirmation before treating it as final. A production system may need a
stronger detector; the spec leaves that decision open.

Tracking (Section 4.2, "same person across frames") is implemented
separately in ai/tracking.py-equivalent logic inside suspicious_activity.py
and virtual_fence.py via simple centroid tracking, since the spec only
requires that identity be maintained across frames -- it doesn't mandate
a specific tracking algorithm (e.g. SORT/DeepSORT) either.
"""

import cv2

_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_humans(bgr_frame):
    """
    Returns a list of detections:
        {"object": "person", "confidence": float, "x1": int, "y1": int, "x2": int, "y2": int}
    matching the fields listed in Section 4.1.
    """
    rects, weights = _hog.detectMultiScale(
        bgr_frame, winStride=(4, 4), padding=(8, 8), scale=1.05
    )
    detections = []
    for (x, y, w, h), weight in zip(rects, weights):
        detections.append({
            "object": "person",
            "confidence": float(weight),
            "x1": int(x), "y1": int(y),
            "x2": int(x + w), "y2": int(y + h),
        })
    return detections


class CentroidTracker:
    """
    Minimal centroid tracker so the same person keeps the same ID across
    frames (Section 4.2). Deliberately simple -- spec asks only that
    identity persist across frames, not for a specific algorithm.
    """
    def __init__(self, max_distance=80):
        self.next_id = 0
        self.objects = {}  # id -> (cx, cy)
        self.max_distance = max_distance

    @staticmethod
    def _centroid(det):
        return ((det["x1"] + det["x2"]) / 2.0, (det["y1"] + det["y2"]) / 2.0)

    def update(self, detections):
        """Returns list of (track_id, detection)."""
        assigned = []
        used_ids = set()
        for det in detections:
            cx, cy = self._centroid(det)
            best_id, best_dist = None, self.max_distance
            for oid, (ox, oy) in self.objects.items():
                if oid in used_ids:
                    continue
                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = oid, dist
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            self.objects[best_id] = (cx, cy)
            used_ids.add(best_id)
            assigned.append((best_id, det))
        return assigned
