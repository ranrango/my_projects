import cv2
import numpy as np


# Colour palette for consistent per-identity colouring
_PALETTE = [
    (0, 255, 0),    # green  — granted
    (0, 0, 255),    # red    — denied / unknown
    (255, 165, 0),  # orange — liveness fail
    (255, 255, 0),  # yellow — locked out
]


def color_for(decision: str) -> tuple:
    if decision == "GRANT":
        return (0, 255, 0)
    if decision == "LOCKED":
        return (0, 200, 255)
    if decision == "LIVENESS_FAIL":
        return (0, 165, 255)
    return (0, 0, 255)


def draw_face(frame: np.ndarray, bbox: tuple, identity: str, score: float, decision: str) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    col = color_for(decision)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    label = f"{identity} ({score:.2f}) [{decision}]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_gate_status(frame: np.ndarray, gate_open: bool) -> np.ndarray:
    text = "GATE: OPEN" if gate_open else "GATE: CLOSED"
    col = (0, 255, 0) if gate_open else (0, 0, 255)
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return frame
