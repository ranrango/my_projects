from __future__ import annotations

import os
import cv2
import numpy as np

from face_gate.recognition.embedder import FaceEmbedder
from face_gate.detection.detector import FaceDetector


class GalleryIndex:
    """In-memory face gallery with cosine-similarity search."""

    def __init__(self):
        self.embeddings: list[np.ndarray] = []
        self.labels: list[str] = []

    def add(self, embedding: np.ndarray, label: str):
        self.embeddings.append(embedding)
        self.labels.append(label)

    def search(self, query: np.ndarray, threshold: float = 0.55) -> tuple[str, float]:
        """Return (best_label, score). Returns ('UNKNOWN', score) if below threshold."""
        if not self.embeddings:
            return "UNKNOWN", 0.0
        matrix = np.stack(self.embeddings)
        scores = matrix @ query
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < threshold:
            return "UNKNOWN", best_score
        return self.labels[best_idx], best_score

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=np.array(self.embeddings),
            labels=np.array(self.labels),
        )

    @classmethod
    def load(cls, path: str) -> "GalleryIndex":
        idx = cls()
        if not os.path.exists(path):
            return idx
        data = np.load(path, allow_pickle=True)
        idx.embeddings = list(data["embeddings"])
        idx.labels = list(data["labels"])
        return idx

    def __len__(self):
        return len(self.embeddings)


class Recognizer:
    def __init__(self, match_threshold: float = 0.55, unknown_label: str = "UNKNOWN"):
        self.embedder = FaceEmbedder()
        self.gallery = GalleryIndex()
        self.match_threshold = match_threshold
        self.unknown_label = unknown_label

    def load_gallery(self, index_path: str):
        self.gallery = GalleryIndex.load(index_path)

    def identify(self, face_crop: np.ndarray) -> tuple[str, float]:
        emb = self.embedder.embed(face_crop)
        return self.gallery.search(emb, self.match_threshold)


def build_gallery_from_dir(
    faces_dir: str,
    index_path: str,
    min_images: int = 1,
) -> GalleryIndex:
    """Scan faces_dir/<identity>/*.jpg and build a gallery index.

    Each sub-directory name is treated as the person's identity label.
    """
    detector = FaceDetector()
    embedder = FaceEmbedder()
    gallery = GalleryIndex()

    if not os.path.isdir(faces_dir):
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")

    enrolled = 0
    skipped = 0
    for identity in sorted(os.listdir(faces_dir)):
        person_dir = os.path.join(faces_dir, identity)
        if not os.path.isdir(person_dir):
            continue
        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(images) < min_images:
            print(f"  [SKIP] {identity}: only {len(images)} images (need {min_images})")
            skipped += 1
            continue

        embeddings_for_person = []
        for fname in images:
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            faces = detector.detect(img)
            if not faces:
                emb = embedder.embed(cv2.resize(img, (112, 112)))
            else:
                crop = detector.crop_face(img, faces[0]["bbox"])
                emb = embedder.embed(crop)
            embeddings_for_person.append(emb)

        if not embeddings_for_person:
            skipped += 1
            continue

        mean_emb = np.mean(embeddings_for_person, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-6:
            mean_emb /= norm
        gallery.add(mean_emb, identity)
        enrolled += 1
        print(f"  [OK] {identity}: enrolled from {len(embeddings_for_person)} images")

    gallery.save(index_path)
    print(f"\nGallery built: {enrolled} enrolled, {skipped} skipped → {index_path}")
    return gallery
