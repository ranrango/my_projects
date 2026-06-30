from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from reid_baseline.model import ReIDModel


def build_inference_transform(image_size=(256, 128)):
    return transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


class ReIDInferencer:
    def __init__(self, checkpoint_path, device="auto", image_size=(256, 128)):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.transform = build_inference_transform(image_size)
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model(self.checkpoint_path)

    def _load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            num_classes = checkpoint.get("num_classes", 751)
            embedding_dim = checkpoint.get("embedding_dim", 512)
        else:
            state = checkpoint
            classifier_weight = state.get("classifier.weight")
            embedding_weight = state.get("embedding.weight")
            num_classes = classifier_weight.shape[0] if classifier_weight is not None else 751
            embedding_dim = embedding_weight.shape[0] if embedding_weight is not None else 512

        model = ReIDModel(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def extract_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self.extract_tensor(tensor)[0]

    def extract_batch(self, image_paths, batch_size=64):
        features = []
        paths = [str(Path(path)) for path in image_paths]
        with torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start:start + batch_size]
                images = [
                    self.transform(Image.open(path).convert("RGB"))
                    for path in batch_paths
                ]
                tensor = torch.stack(images, dim=0).to(self.device)
                features.append(self.extract_tensor(tensor))
        if not features:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(features)

    def extract_tensor(self, tensor):
        with torch.no_grad():
            features, _ = self.model(tensor)
            features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype(np.float32)


class GalleryIndex:
    def __init__(self, features, paths, ids=None, metadata=None):
        self.features = features.astype(np.float32)
        self.paths = np.asarray(paths)
        self.ids = np.asarray(ids if ids is not None else [Path(path).stem for path in paths])
        self.metadata = metadata or {}
        if len(self.paths) != len(self.features):
            raise ValueError("features 和 paths 数量不一致。")
        if len(self.ids) != len(self.features):
            raise ValueError("features 和 ids 数量不一致。")

    @classmethod
    def load(cls, index_path):
        data = np.load(index_path, allow_pickle=True)
        metadata = data["metadata"].item() if "metadata" in data else {}
        ids = data["ids"] if "ids" in data else None
        return cls(data["features"], data["paths"], ids=ids, metadata=metadata)

    def save(self, index_path):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            index_path,
            features=self.features,
            paths=self.paths,
            ids=self.ids,
            metadata=np.asarray(self.metadata, dtype=object),
        )

    def search(self, query_feature, topk=5, threshold=None, include_path=True):
        if self.features.size == 0:
            return []
        query = query_feature.astype(np.float32)
        if query.ndim == 1:
            query = query[None, :]
        scores = np.matmul(query, self.features.T)[0]
        limit = min(max(int(topk), 1), len(scores))
        candidates = np.argpartition(-scores, limit - 1)[:limit]
        order = candidates[np.argsort(-scores[candidates])]

        results = []
        for rank, index in enumerate(order):
            score = float(scores[index])
            if threshold is not None and score < threshold:
                continue
            item = {
                "rank": int(rank + 1),
                "id": str(self.ids[index]),
                "score": score,
            }
            if include_path:
                item["path"] = str(self.paths[index])
            results.append(item)
        return results
