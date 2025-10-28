import cv2


def get_haar_cascade():
    global _CACHED_CASCADE
    try:
        _CACHED_CASCADE
    except NameError:
        _CACHED_CASCADE = None
    if _CACHED_CASCADE is None:
        _CACHED_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CACHED_CASCADE


def detect_largest_face_bgr(frame_bgr, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    detector = get_haar_cascade()
    faces = detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return x, y, w, h


def crop_face_square(frame_bgr, bbox, padding=0.25):
    if bbox is None:
        return None
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    cx = x + bw // 2
    cy = y + bh // 2
    side = int(max(bw, bh) * (1 + padding))
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return frame_bgr[y1:y2, x1:x2]
