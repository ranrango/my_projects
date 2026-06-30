import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

from reid_baseline.market1501_loader import Market1501Dataset
from reid_baseline.model import ReIDModel
from reid_baseline.utils import load_config, merge_overrides

# ==== 特征提取 ====

def extract_features(model, loader, device):
    model.eval()
    features, labels, camids = [], [], []

    with torch.no_grad():
        for imgs, pids, cams in tqdm(loader):
            imgs = imgs.to(device)
            feats, _ = model(imgs)
            feats = feats.cpu().numpy()
            features.append(feats)
            labels.extend(pids)
            camids.extend(cams)

    features = np.vstack(features)
    labels = np.array(labels)
    camids = np.array(camids)
    return features, labels, camids

# ==== 距离计算 + 检索评估 ====

def compute_metrics(query_feat, query_label, query_cam, gallery_feat, gallery_label, gallery_cam):
    # 使用欧氏距离的平方，避免构造 query x gallery x dim 的巨大三维数组。
    query_norm = np.sum(np.square(query_feat), axis=1, keepdims=True)
    gallery_norm = np.sum(np.square(gallery_feat), axis=1, keepdims=True).T
    dist_mat = query_norm + gallery_norm - 2 * np.matmul(query_feat, gallery_feat.T)
    dist_mat = np.maximum(dist_mat, 0.0)

    num_q, num_g = dist_mat.shape
    cmc = np.zeros(num_g)
    ap = 0.0
    valid_queries = 0

    for i in range(num_q):
        q_label = query_label[i]
        q_cam = query_cam[i]

        order = np.argsort(dist_mat[i])
        remove = (gallery_label == q_label) & (gallery_cam == q_cam)  # 去除同摄像头的同 ID
        keep = np.invert(remove)

        match = (gallery_label[order] == q_label).astype(int)[keep[order]]

        if match.sum() == 0:
            continue  # 无有效匹配

        # CMC
        first_index = np.where(match == 1)[0][0]
        cmc[first_index:] += 1

        # AP
        true = match
        score = -dist_mat[i][order][keep[order]]
        ap += average_precision_score(true, score)
        valid_queries += 1

    if valid_queries == 0:
        raise RuntimeError("没有有效 query：请检查 query/gallery 是否包含跨摄像头的同一 PID。")

    cmc = cmc / valid_queries
    mAP = ap / valid_queries
    return cmc, mAP

# ==== 主程序入口 ====

def build_transform(image_size):
    transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Market1501 ReID checkpoint.")
    parser.add_argument("--config", default="reid_baseline/configs/reid_config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument("--max-query", type=int, default=None, help="只评估前 N 张 query 图")
    parser.add_argument("--max-gallery", type=int, default=None, help="只评估前 N 张 gallery 图")
    return parser.parse_args()


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint
    return {"model_state": checkpoint}


def main():
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), {
        "data.root": args.data_root,
        "model.checkpoint": args.checkpoint,
        "train.batch_size": args.batch_size,
        "train.num_workers": args.num_workers,
        "runtime.device": args.device,
    })

    device_name = cfg.get("runtime", {}).get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint = load_checkpoint(cfg["model"]["checkpoint"], device)
    num_classes = checkpoint.get("num_classes", cfg["data"]["num_classes"])
    embedding_dim = checkpoint.get("embedding_dim", cfg["model"]["embedding_dim"])

    transform = build_transform(cfg["data"]["image_size"])

    query_set = Market1501Dataset(
        cfg["data"]["root"],
        mode='query',
        transform=transform,
        max_samples=args.max_query,
    )
    gallery_set = Market1501Dataset(
        cfg["data"]["root"],
        mode='gallery',
        transform=transform,
        max_samples=args.max_gallery,
    )

    query_loader = DataLoader(
        query_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = ReIDModel(num_classes=num_classes, embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    print(f"使用设备: {device}")

    print("提取 query 特征...")
    q_feat, q_label, q_cam = extract_features(model, query_loader, device)

    print("提取 gallery 特征...")
    g_feat, g_label, g_cam = extract_features(model, gallery_loader, device)

    print("计算检索指标...")
    cmc, mAP = compute_metrics(q_feat, q_label, q_cam, g_feat, g_label, g_cam)

    print("======= Evaluation Results =======")
    print(f"mAP: {mAP:.4f}")
    for r in [1, 5, 10]:
        if r <= len(cmc):
            print(f"Rank-{r}: {cmc[r-1]:.4f}")

if __name__ == '__main__':
    main()
