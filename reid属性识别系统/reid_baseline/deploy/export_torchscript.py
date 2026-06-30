import argparse
from pathlib import Path

import torch

from reid_baseline.deploy.inferencer import ReIDInferencer


class FeatureOnlyModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features, _ = self.model(x)
        return torch.nn.functional.normalize(features, p=2, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Export a ReID checkpoint to TorchScript.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="deploy/reid_feature_extractor.pt")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    inferencer = ReIDInferencer(args.checkpoint, device=args.device)
    model = FeatureOnlyModel(inferencer.model).eval()
    example = torch.randn(1, 3, 256, 128, device=inferencer.device)
    traced = torch.jit.trace(model, example)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)
    print(f"TorchScript 模型已导出: {args.output}")


if __name__ == "__main__":
    main()

