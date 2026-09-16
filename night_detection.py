"""
ai/night_detection.py

Implements Module H (Section 10) -- Night-Time Movement Detection.

The spec gives only the objective (identify movement during night-time
surveillance), no method. DEMO CHOICE (flagged): treat a frame as
"night" when its average grayscale brightness is below a threshold, and
report "night-time movement" when that coincides with any human/vehicle
detection already found in the frame this cycle.
"""

import cv2
import numpy as np

NIGHT_BRIGHTNESS_THRESHOLD = 60  # out of 255; OUR choice, not spec-mandated


def is_night_frame(bgr_frame):
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < NIGHT_BRIGHTNESS_THRESHOLD, mean_brightness


def night_movement_event(bgr_frame, has_motion_detections: bool):
    night, brightness = is_night_frame(bgr_frame)
    triggered = night and has_motion_detections
    return {"night": night, "brightness": brightness, "night_movement": triggered}
