# src/emotion_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Facial Emotion Detection using ResNet CNN (pure PyTorch)
#
# Changes from v1:
#   - Runs until user presses ENTER (not 5 second timeout)
#   - Bounding box drawn around detected face
#   - Emotion counts accumulate in background across all frames
#   - On tie: user is prompted to pick from tied emotions
#   - capture_emotion() is the single function called by run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import numpy as np
import torch
import cv2
from collections import Counter
from typing import Optional, Tuple

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.40   # 0.0–1.0
DETECTION_STRIDE = 2
ROI_PADDING_RATIO = 0.10
PROBS_EMA_ALPHA = 0.55
MIN_STABLE_FRAMES = 1

BOX_EMA_ALPHA = 0.25
BOX_HOLD_FRAMES = 10
BOX_IOU_HINT_THRESHOLD = 0.15
RECENT_WINDOW_SIZE = 28

VALID_EMOTIONS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

_PROJECT_ROOT       = os.path.dirname(_SRC)
_DEFAULT_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "models", "emotion", "resnet_emotion.pth"
)

_NUM_CLASSES        = 7
_IN_CHANNELS        = 3
_LINEAR_IN_FEATURES = 2048
_IMG_SIZE           = 44

_emotion_model = None
_face_cascade  = None
_device        = None
_clahe         = None


# ── Model loading ─────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[emotion_detector] Device: {_device}")
    return _device


def _load_emotion_model(model_path: str = _DEFAULT_MODEL_PATH):
    global _emotion_model
    if _emotion_model is not None:
        return _emotion_model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[emotion_detector] Model not found:\n  {model_path}"
        )
    try:
        from inference import load_model
    except ImportError:
        raise ImportError(
            "[emotion_detector] inference.py not found in src/."
        )
    print("[emotion_detector] Loading ResNet emotion model ...")
    _emotion_model = load_model(
        model_path         = model_path,
        num_classes        = _NUM_CLASSES,
        in_channels        = _IN_CHANNELS,
        linear_in_features = _LINEAR_IN_FEATURES,
        device             = _get_device(),
    )
    print("[emotion_detector] ResNet emotion model ready.")
    return _emotion_model


def _load_face_cascade():
    global _face_cascade
    if _face_cascade is not None:
        return _face_cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(cascade_path)
    if _face_cascade.empty():
        raise RuntimeError("[emotion_detector] Haar cascade failed to load.")
    return _face_cascade


def _load_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _clahe


def _box_area(box: Tuple[int, int, int, int]) -> int:
    _, _, w, h = box
    return max(0, int(w)) * max(0, int(h))


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    union = _box_area(a) + _box_area(b) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _smooth_box(
    prev_box: Tuple[int, int, int, int],
    new_box: Tuple[int, int, int, int],
    alpha: float,
) -> Tuple[int, int, int, int]:
    px, py, pw, ph = prev_box
    nx, ny, nw, nh = new_box
    inv = 1.0 - alpha
    return (
        int(round((px * inv) + (nx * alpha))),
        int(round((py * inv) + (ny * alpha))),
        int(round((pw * inv) + (nw * alpha))),
        int(round((ph * inv) + (nh * alpha))),
    )


def _filter_face_boxes(
    frame_shape: Tuple[int, int, int],
    boxes: list[Tuple[int, int, int, int]],
) -> list[Tuple[int, int, int, int]]:
    h, w = frame_shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return boxes

    keep: list[Tuple[int, int, int, int]] = []
    for x, y, bw, bh in boxes:
        if bw <= 0 or bh <= 0:
            continue
        area = float(bw * bh) / frame_area
        aspect = float(bw) / float(bh)
        if area < 0.003:
            continue
        if area > 0.75:
            continue
        if aspect < 0.55 or aspect > 1.80:
            continue
        keep.append((int(x), int(y), int(bw), int(bh)))
    return keep


def _select_face_box(
    boxes: list[Tuple[int, int, int, int]],
    hint_box: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None

    if hint_box is not None:
        best_iou = -1.0
        best = None
        for b in boxes:
            iou = _box_iou(hint_box, b)
            if iou > best_iou:
                best_iou = iou
                best = b
        if best is not None and best_iou >= BOX_IOU_HINT_THRESHOLD:
            return best

    return max(boxes, key=_box_area)


def _crop_face_with_padding(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    padding_ratio: float,
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """Crop face ROI with contextual padding while staying in frame bounds."""
    x, y, w, h = box
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None, (x, y, w, h)

    padded_box = (x1, y1, x2 - x1, y2 - y1)
    return frame[y1:y2, x1:x2], padded_box


# ── Single frame detection ────────────────────────────────────────────────────

def detect_emotion(
    frame: np.ndarray,
    return_probs: bool = False,
    clahe=None,
    hint_box: Optional[Tuple[int, int, int, int]] = None,
) -> tuple:
    """
    Detect emotion from one BGR frame.
    Returns (emotion_str, confidence_float, face_box_or_None[, probs]).
    face_box is (x, y, w, h) of the largest detected face, or None.
    Never raises.
    """
    if frame is None or frame.size == 0:
        if return_probs:
            return "neutral", 0.0, None, None
        return "neutral", 0.0, None

    try:
        from inference import preprocess_roi, EMOTION_LABELS

        model   = _load_emotion_model()
        cascade = _load_face_cascade()
        device  = _get_device()

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            if return_probs:
                return "neutral", 0.0, None, None
            return "neutral", 0.0, None

        boxes = _filter_face_boxes(
            frame.shape,
            [tuple(map(int, f)) for f in faces],
        )
        box = _select_face_box(boxes, hint_box)
        if box is None:
            if return_probs:
                return "neutral", 0.0, None, None
            return "neutral", 0.0, None

        face_roi, _roi_box = _crop_face_with_padding(
            frame,
            box,
            padding_ratio=ROI_PADDING_RATIO,
        )

        if face_roi is None or face_roi.size == 0:
            if return_probs:
                return "neutral", 0.0, None, None
            return "neutral", 0.0, None

        tensor = preprocess_roi(
            roi_bgr=face_roi,
            img_size=_IMG_SIZE,
            in_channels=_IN_CHANNELS,
            imagenet_norm=True,
            clahe=clahe,
        ).to(device)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        emotion = EMOTION_LABELS[idx].lower().strip()
        confidence = float(probs[idx])

        if emotion not in VALID_EMOTIONS:
            if return_probs:
                return "neutral", 0.0, None, probs
            return "neutral", 0.0, None
        if confidence < CONFIDENCE_THRESHOLD:
            if return_probs:
                return "neutral", 0.0, box, probs
            return "neutral", 0.0, box

        if return_probs:
            return emotion, confidence, box, probs
        return emotion, confidence, box

    except Exception:
        if return_probs:
            return "neutral", 0.0, None, None
        return "neutral", 0.0, None


# ── Tie resolution ────────────────────────────────────────────────────────────

def _resolve_tie(tied_emotions: list, counts: dict) -> str:
    """
    When two or more emotions have equal top counts, ask the user to pick.
    Shows the tied emotions and their counts clearly.
    """
    print("\n" + "─" * 48)
    print("  Tie detected between emotions:")
    for em in tied_emotions:
        print(f"    {em:<12} — detected {counts[em]} times")
    print("\n  Which emotion should set the voice tone?")
    for i, em in enumerate(tied_emotions, 1):
        print(f"    [{i}] {em}")

    while True:
        raw = input("  Your choice (number) > ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(tied_emotions):
                chosen = tied_emotions[idx]
                print(f"  ✓  You chose: {chosen}")
                return chosen
        print(f"  ⚠  Enter a number between 1 and {len(tied_emotions)}")


# ── Main capture function (called by run_pipeline.py) ─────────────────────────

def capture_emotion() -> str:
    """
    Opens the webcam and runs continuous emotion detection until the user
    presses ENTER. Emotion counts accumulate silently in the background.
    A bounding box is drawn around the face on the live feed.

    On exit:
    - If one emotion has the highest count → returns it automatically.
    - If there is a tie → prompts user to choose from tied options.
    - If no confident detections → falls back to manual text input.

    Returns:
        emotion: str — one of the 7 valid SentiSign labels.
    """
    print("\n" + "─" * 64)
    print("  [Emotion Detection]  Webcam opening ...")
    print("  Express your emotion naturally. Counts accumulate in background.")
    print("  Press ENTER in this terminal when ready to continue.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠  Webcam unavailable → manual input.")
        return _manual_fallback()

    # Emotion counter — accumulates across entire session
    emotion_counts: Counter = Counter()
    emotion_scores: Counter = Counter()
    recent_items: list[Tuple[str, float]] = []
    recent_scores: Counter = Counter()
    frame_idx = 0
    smoothed_probs = None
    stable_emotion = None
    stable_hits = 0
    tracked_box = None
    box_hold = 0

    # Run detection in a loop; stop when user presses ENTER
    # We use cv2.waitKey(1) for the window and check a flag for ENTER.
    import threading
    enter_pressed = threading.Event()

    def _wait_for_enter():
        input()   # blocks until ENTER
        enter_pressed.set()

    t = threading.Thread(target=_wait_for_enter, daemon=True)
    t.start()

    clahe = _load_clahe()

    while not enter_pressed.is_set():
        ret, frame = cap.read()
        if not ret:
            print("  ⚠  Webcam read failed.")
            break

        # Run detection every nth frame to reduce load while keeping updates smooth.
        if frame_idx % DETECTION_STRIDE == 0:
            _emotion, _confidence, detected_box, probs = detect_emotion(
                frame,
                return_probs=True,
                clahe=clahe,
                hint_box=tracked_box,
            )

            if detected_box is not None:
                if tracked_box is None:
                    tracked_box = detected_box
                else:
                    tracked_box = _smooth_box(tracked_box, detected_box, BOX_EMA_ALPHA)
                box_hold = BOX_HOLD_FRAMES
            elif box_hold > 0:
                box_hold -= 1
            else:
                tracked_box = None

            if probs is not None:
                if smoothed_probs is None:
                    smoothed_probs = probs
                else:
                    smoothed_probs = (PROBS_EMA_ALPHA * probs) + ((1.0 - PROBS_EMA_ALPHA) * smoothed_probs)

                smooth_idx = int(np.argmax(smoothed_probs))
                smooth_emotion = VALID_EMOTIONS[smooth_idx]
                smooth_conf = float(smoothed_probs[smooth_idx])

                if smooth_conf >= CONFIDENCE_THRESHOLD and tracked_box is not None:
                    if stable_emotion == smooth_emotion:
                        stable_hits += 1
                    else:
                        stable_emotion = smooth_emotion
                        stable_hits = 1

                    if stable_hits >= MIN_STABLE_FRAMES:
                        emotion_counts[smooth_emotion] += 1
                        emotion_scores[smooth_emotion] += smooth_conf
                        recent_items.append((smooth_emotion, smooth_conf))
                        recent_scores[smooth_emotion] += smooth_conf

                        if len(recent_items) > RECENT_WINDOW_SIZE:
                            old_em, old_conf = recent_items.pop(0)
                            recent_scores[old_em] -= old_conf
                            if recent_scores[old_em] <= 1e-6:
                                del recent_scores[old_em]
                else:
                    stable_hits = 0

        # ── Draw bounding box ─────────────────────────────────────────────────
        if tracked_box is not None:
            x, y, w, h = tracked_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ── Overlay: current top emotion + total detections ───────────────────
        if smoothed_probs is not None:
            now_idx = int(np.argmax(smoothed_probs))
            now_emotion = VALID_EMOTIONS[now_idx]
            now_conf = float(smoothed_probs[now_idx])
        else:
            now_emotion, now_conf = "...", 0.0

        if recent_scores:
            recent_emotion, recent_score = recent_scores.most_common(1)[0]
        else:
            recent_emotion, recent_score = "...", 0.0

        total = sum(emotion_counts.values())

        cv2.putText(frame,
            f"Now: {now_emotion} ({now_conf:.2f})  |  Recent: {recent_emotion}  |  Reads: {total}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame,
            "Press ENTER in terminal when done",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        cv2.imshow("SentiSign  |  Emotion Detection", frame)
        cv2.waitKey(1)   # 1ms poll — keeps window responsive

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ── No confident detections at all ───────────────────────────────────────
    if not emotion_counts:
        print("  ⚠  No confident detections → manual input.")
        return _manual_fallback()

    # ── Print final counts ────────────────────────────────────────────────────
    print("\n  Emotion detection complete. Results:")
    print("  " + "─" * 36)
    for em, count in emotion_counts.most_common():
        bar = "█" * min(count, 40)
        print(f"  {em:<12} {count:>4}  {bar}")
    print("  " + "─" * 36)

    # ── Determine winner ──────────────────────────────────────────────────────
    # Prefer the most recent window so the selected emotion reflects what the user
    # is doing near the end of the capture (more responsive than lifetime totals).
    score_source = recent_scores if recent_scores else emotion_scores
    top_score = score_source.most_common(1)[0][1]
    top_emotions = [em for em, score in score_source.items() if abs(score - top_score) < 0.10]

    if len(top_emotions) == 1:
        winner = top_emotions[0]
        print(
            f"\n  ✓  Detected emotion: {winner}  "
            f"({emotion_counts[winner]} stable readings, score {float(score_source[winner]):.2f})"
        )
        return winner
    else:
        # Tie — let user decide
        return _resolve_tie(sorted(top_emotions), dict(emotion_counts))


def _manual_fallback() -> str:
    """Ask user to type emotion when webcam detection fails or produces no data."""
    print(f"  Supported emotions: {', '.join(VALID_EMOTIONS)}")
    while True:
        em = input("  Emotion > ").strip().lower()
        if em in VALID_EMOTIONS:
            print(f"  ✓  Emotion: {em}")
            return em
        print(f"  ⚠  '{em}' not recognised. Options: {', '.join(VALID_EMOTIONS)}")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  ResNet Emotion Detector — Standalone Test")
    print("  Express different emotions at the camera.")
    print("  Press ENTER in this terminal when done.")
    print("=" * 64 + "\n")

    result = capture_emotion()
    print(f"\n  Final emotion selected: {result}\n")
