"""
ai/anpr.py

Implements Module E (Section 7) -- Automatic Number Plate Recognition.

Workflow per Section 7.1: Vehicle -> Plate Detection -> Plate Region ->
Plate Recognition (OCR) -> Plate Information.

SPEC GAP: the spec names no specific plate-detection or OCR technology.
DEMO CHOICE (flagged): OpenCV's bundled haarcascade_russian_plate_number.xml
for plate LOCALIZATION, and Tesseract OCR (pytesseract) for reading the
characters. Both are already available offline. Confirm before treating
as final -- a production ANPR system usually needs a purpose-trained
plate detector for accuracy.
"""

import cv2
import pytesseract

PLATE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"


def detect_plate_regions(bgr_frame):
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(PLATE_CASCADE_PATH)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 20))
    return list(plates)


def read_plate_text(bgr_frame, box):
    x, y, w, h = box
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    plate_roi = gray[y:y + h, x:x + w]
    # upscale small plate crops -- improves OCR reliability
    plate_roi = cv2.resize(plate_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, plate_roi = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        plate_roi, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )
    return text.strip()


def run_anpr(bgr_frame):
    """
    Returns a list of dicts:
        {"box": (x,y,w,h), "plate_text": "<string or ''>"}
    matching Section 7.2's "plate information" output.
    """
    results = []
    for box in detect_plate_regions(bgr_frame):
        text = read_plate_text(bgr_frame, box)
        results.append({"box": tuple(int(v) for v in box), "plate_text": text})
    return results
