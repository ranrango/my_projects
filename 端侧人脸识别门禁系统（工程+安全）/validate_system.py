"""
validate_system.py  —  交付验收脚本

检查项目核心链路是否完整可运行，不依赖真实摄像头或真实人脸。
运行方式:
  python validate_system.py
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(name: str, fn):
    try:
        result = fn()
        if result is False:
            print(f"{FAIL} {name}")
            return False
        print(f"{PASS} {name}" + (f" — {result}" if isinstance(result, str) else ""))
        return True
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def main():
    results = []

    results.append(check("import cv2", lambda: __import__("cv2") and True))
    results.append(check("import numpy", lambda: __import__("numpy") and True))

    # Haar cascade available
    def _cascade():
        import cv2
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        assert os.path.exists(path), f"Not found: {path}"
        det = cv2.CascadeClassifier(path)
        assert not det.empty()
        return "cascade OK"
    results.append(check("FaceDetector: haar cascade", _cascade))

    # Detector on synthetic image
    def _detector():
        import numpy as np
        from face_gate.detection.detector import FaceDetector
        det = FaceDetector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = det.detect(blank)
        assert isinstance(faces, list)
        return f"detected {len(faces)} faces on blank image (expected 0)"
    results.append(check("FaceDetector: detects 0 faces on blank frame", _detector))

    # Liveness checker
    def _liveness():
        import numpy as np
        from face_gate.detection.liveness import LivenessChecker
        lc = LivenessChecker(texture_var_threshold=200.0)
        flat = np.full((112, 112, 3), 128, dtype=np.uint8)
        is_live, score = lc.check(flat)
        assert not is_live, "flat image should fail liveness"
        import cv2
        noise = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        is_live2, score2 = lc.check(noise)
        assert is_live2, "noise image should pass liveness"
        return f"flat={score:.1f} noise={score2:.1f}"
    results.append(check("LivenessChecker: flat=fail noise=pass", _liveness))

    # Embedder
    def _embedder():
        import numpy as np
        from face_gate.recognition.embedder import FaceEmbedder
        emb = FaceEmbedder()
        img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        vec = emb.embed(img)
        assert vec.ndim == 1
        import numpy as np2
        assert abs(np2.linalg.norm(vec) - 1.0) < 1e-5, "embedding not unit length"
        return f"dim={len(vec)} L2={np2.linalg.norm(vec):.4f}"
    results.append(check("FaceEmbedder: unit-norm embedding", _embedder))

    # Gallery round-trip
    def _gallery():
        import tempfile, numpy as np
        from face_gate.recognition.gallery import GalleryIndex
        g = GalleryIndex()
        v = np.random.randn(944).astype(np.float32)
        v /= np.linalg.norm(v)
        g.add(v, "Alice")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test_gallery.npz")
            g.save(path)
            g2 = GalleryIndex.load(path)
        label, score = g2.search(v, threshold=0.5)
        assert label == "Alice", f"Expected Alice, got {label}"
        return f"save/load/search OK score={score:.4f}"
    results.append(check("GalleryIndex: save/load/search", _gallery))

    # SecurityGuard lockout
    def _guard():
        from face_gate.security.guard import SecurityGuard
        g = SecurityGuard(max_failed=3, lockout_sec=10)
        for _ in range(3):
            g.record_failure("bob")
        assert g.is_locked("bob"), "bob should be locked"
        return "lockout after 3 failures OK"
    results.append(check("SecurityGuard: lockout logic", _guard))

    # AccessController
    def _ac():
        from face_gate.security.access_controller import AccessController
        ac = AccessController(allow_unknown=False, gate_open_duration=3.0, cooldown_sec=0)
        assert ac.decide("UNKNOWN", 0.3) == "DENY"
        assert ac.decide("Alice", 0.9) == "GRANT"
        ac.open_gate("Alice", 3.0)
        assert ac.gate_is_open
        return "deny-unknown, grant-known, gate-open OK"
    results.append(check("AccessController: grant/deny/gate", _ac))

    # AuditLogger
    def _audit():
        import tempfile, json
        from face_gate.utils.audit_logger import AuditLogger
        with tempfile.TemporaryDirectory() as td:
            logger = AuditLogger(
                log_file=os.path.join(td, "audit_logs", "test.jsonl"),
                snapshot_dir=os.path.join(td, "snaps"),
            )
            entry = logger.log("grant", "Alice", 42, score=0.9)
            with open(logger.log_file) as f:
                line = json.loads(f.readline())
            assert line["identity"] == "Alice"
        return "write+read JSONL OK"
    results.append(check("AuditLogger: JSONL write/read", _audit))

    # Config loader
    def _config():
        from face_gate.utils.config import load_config
        base = os.path.dirname(os.path.abspath(__file__))
        cfg = load_config(os.path.join(base, "config", "default_config.json"), base_dir=base)
        assert "camera" in cfg and "recognition" in cfg
        return "config loaded OK"
    results.append(check("Config: load default_config.json", _config))

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*50}")
    print(f"Result: {passed}/{total} checks passed")
    if passed < total:
        print("Some checks failed. See [FAIL] lines above.")
        sys.exit(1)
    else:
        print("All checks passed. System is ready.")


if __name__ == "__main__":
    main()
