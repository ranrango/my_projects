import argparse
from pathlib import Path

from tqdm import tqdm

from reid_baseline.deploy.inferencer import GalleryIndex, ReIDInferencer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a ReID gallery feature index.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery-dir", required=True)
    parser.add_argument("--output", default="deploy/gallery_index.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--id-mode",
        choices=["stem", "relative_path", "absolute_path"],
        default="relative_path",
        help="图库条目的业务 ID 生成方式。",
    )
    parser.add_argument(
        "--skip-invalid-market1501",
        action="store_true",
        help="跳过 Market1501 中 pid<=0 的干扰图。",
    )
    return parser.parse_args()


def is_valid_market1501_image(path):
    try:
        pid = int(path.name.split("_")[0])
    except (ValueError, IndexError):
        return True
    return pid > 0


def collect_images(root, skip_invalid_market1501=False):
    root = Path(root)
    images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if skip_invalid_market1501:
        images = [path for path in images if is_valid_market1501_image(path)]
    return images


def make_ids(image_paths, gallery_dir, mode):
    gallery_dir = Path(gallery_dir)
    ids = []
    for path in image_paths:
        if mode == "stem":
            ids.append(path.stem)
        elif mode == "absolute_path":
            ids.append(str(path))
        else:
            ids.append(str(path.relative_to(gallery_dir)))
    return ids


def main():
    args = parse_args()
    image_paths = collect_images(
        args.gallery_dir,
        skip_invalid_market1501=args.skip_invalid_market1501,
    )
    if not image_paths:
        raise RuntimeError(f"图库目录没有图片: {args.gallery_dir}")

    inferencer = ReIDInferencer(args.checkpoint, device=args.device)
    features = []
    paths = []
    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Build gallery"):
        batch_paths = image_paths[start:start + args.batch_size]
        features.append(inferencer.extract_batch(batch_paths, batch_size=args.batch_size))
        paths.extend(str(path) for path in batch_paths)

    import numpy as np
    index = GalleryIndex(
        features=np.vstack(features),
        paths=paths,
        ids=make_ids(image_paths, args.gallery_dir, args.id_mode),
        metadata={
            "checkpoint": str(Path(args.checkpoint)),
            "gallery_dir": str(Path(args.gallery_dir)),
            "id_mode": args.id_mode,
        },
    )
    index.save(args.output)
    print(f"图库索引已保存: {args.output}")
    print(f"图片数量: {len(paths)}")


if __name__ == "__main__":
    main()
