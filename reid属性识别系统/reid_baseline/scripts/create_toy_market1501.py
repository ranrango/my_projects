import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny Market1501-like dataset.")
    parser.add_argument("--output", default="data/toy_market1501")
    parser.add_argument("--num-pids", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 256], metavar=("W", "H"))
    return parser.parse_args()


def make_image(pid, camid, index, size):
    width, height = size
    base = ((pid * 47) % 255, (pid * 83) % 255, (pid * 121) % 255)
    img = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [width // 4, height // 5, width * 3 // 4, height * 4 // 5],
        outline=(255 - base[0], 255 - base[1], 255 - base[2]),
        width=5,
    )
    draw.text((8, 8), f"pid {pid} c{camid}", fill=(255, 255, 255))
    draw.text((8, 28), f"img {index}", fill=(255, 255, 255))
    return img


def main():
    args = parse_args()
    root = Path(args.output)
    subsets = {
        "bounding_box_train": [1, 2],
        "query": [1],
        "bounding_box_test": [2, 3],
    }
    for subset, camids in subsets.items():
        subset_dir = root / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        for pid in range(1, args.num_pids + 1):
            for index, camid in enumerate(camids, start=1):
                name = f"{pid:04d}_c{camid}s1_000{index:03d}_00.jpg"
                make_image(pid, camid, index, tuple(args.image_size)).save(subset_dir / name)
    print(f"toy Market1501 数据已生成: {root}")


if __name__ == "__main__":
    main()
