import argparse

from reid_baseline.deploy.inferencer import GalleryIndex, ReIDInferencer


def parse_args():
    parser = argparse.ArgumentParser(description="Search similar persons in a ReID gallery.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = ReIDInferencer(args.checkpoint, device=args.device)
    index = GalleryIndex.load(args.index)
    query_feature = inferencer.extract_image(args.query)
    results = index.search(query_feature, topk=args.topk)

    print("======= Search Results =======")
    for item in results:
        print(f"Rank-{item['rank']}: score={item['score']:.4f} path={item['path']}")


if __name__ == "__main__":
    main()

