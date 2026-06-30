import argparse
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from reid_baseline.loss import TripletLoss
from reid_baseline.market1501_loader import Market1501Dataset
from reid_baseline.model import ReIDModel
from reid_baseline.utils import load_config, merge_overrides


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Market1501 ReID baseline.")
    parser.add_argument("--config", default="reid_baseline/configs/reid_config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="只用前 N 张训练图做快速测试")
    parser.add_argument("--max-pids", type=int, default=None, help="只用前 N 个身份做快速测试")
    parser.add_argument("--max-images-per-pid", type=int, default=None, help="每个身份最多使用 N 张训练图")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), {
        "data.root": args.data_root,
        "train.num_epochs": args.epochs,
        "train.batch_size": args.batch_size,
        "train.learning_rate": args.lr,
        "train.num_workers": args.num_workers,
        "runtime.device": args.device,
    })

    device_name = cfg.get("runtime", {}).get("device", "auto")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    transform = build_transform(cfg["data"]["image_size"])
    train_dataset = Market1501Dataset(
        cfg["data"]["root"],
        mode="train",
        transform=transform,
        relabel=True,
        max_samples=args.max_samples,
        max_pids=args.max_pids,
        max_images_per_pid=args.max_images_per_pid,
    )
    num_classes = len(train_dataset.pid2label)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    pretrained = cfg["model"].get("pretrained", True) and not args.no_pretrained
    model = ReIDModel(
        num_classes=num_classes,
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=pretrained,
    ).to(device)
    ce_loss = nn.CrossEntropyLoss()
    tri_loss = TripletLoss(margin=cfg["train"]["margin"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(cfg["train"]["num_epochs"]):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['train']['num_epochs']}")

        for imgs, pids, _ in progress:
            imgs, pids = imgs.to(device), pids.to(device)
            feats, logits = model(imgs)
            loss_ce = ce_loss(logits, pids)
            loss_tri = tri_loss(feats, pids)
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch + 1}/{cfg['train']['num_epochs']}] - Loss: {avg_loss:.4f}")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / Path(cfg["model"]["checkpoint"]).name
    torch.save({
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "embedding_dim": cfg["model"]["embedding_dim"],
        "pid2label": train_dataset.pid2label,
        "config": cfg,
    }, checkpoint_path)
    print(f"模型已保存: {checkpoint_path}")


if __name__ == "__main__":
    main()
