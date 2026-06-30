import cv2
import numpy as np
import os


class FaceEmbedder:
    """LBP-histogram based face embedder — zero deep-learning dependencies.

    Encodes a face crop as a concatenated LBP histogram across a 4×4 grid.
    Final vector is L2-normalised to unit length so cosine similarity
    equals the dot product, making thresholding straightforward.

    Accuracy ceiling: ~70-80 % on Market-like near-frontal faces.
    For production swap this class with an ONNX face-recognition model
    (ArcFace, MobileFaceNet) while keeping the same interface.
    """

    def __init__(self, grid: int = 4):
        self._grid = grid
        # Each of the 16 cells produces 59-bin uniform LBP histogram
        self._dim = grid * grid * 59

    @property
    def embedding_dim(self):
        return self._dim

    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Return L2-normalised embedding vector of length embedding_dim."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (112, 112))
        h, w = resized.shape
        cell_h = h // self._grid
        cell_w = w // self._grid
        hists = []
        for r in range(self._grid):
            for c in range(self._grid):
                cell = resized[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                lbp = self._compute_lbp(cell)
                hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 58))
                hists.append(hist.astype(np.float32))
        vec = np.concatenate(hists)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    @staticmethod
    def _compute_lbp(gray_cell: np.ndarray) -> np.ndarray:
        """Uniform LBP with 8 neighbours and radius 1."""
        h, w = gray_cell.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                        (1, 1), (1, 0), (1, -1), (0, -1)]:
            shifted = np.roll(np.roll(gray_cell, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (gray_cell >= shifted).astype(np.uint8)
        # Map to uniform LBP (0-57) + 1 non-uniform bin (58)
        uniform_map = _build_uniform_map()
        return uniform_map[lbp]


_UNIFORM_MAP_CACHE = None


def _build_uniform_map() -> np.ndarray:
    global _UNIFORM_MAP_CACHE
    if _UNIFORM_MAP_CACHE is not None:
        return _UNIFORM_MAP_CACHE
    m = np.zeros(256, dtype=np.uint8)
    uniform_idx = 0
    for i in range(256):
        bits = bin(i).count("1")
        transitions = bin((i ^ ((i << 1) | (i >> 7)) & 0xFF)).count("1")
        if transitions <= 2:
            m[i] = uniform_idx
            uniform_idx += 1
        else:
            m[i] = 58  # non-uniform bin
    _UNIFORM_MAP_CACHE = m
    return m
