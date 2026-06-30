import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

class Market1501Dataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode='train',
        transform=None,
        relabel=False,
        max_samples=None,
        max_pids=None,
        max_images_per_pid=None,
    ):
        """
        :param root_dir: Market1501 根目录（应包含 bounding_box_train 等文件夹）
        :param mode: 'train', 'query', or 'gallery'
        :param transform: 图像预处理（如 Resize, Normalize）
        :param relabel: 训练时将原始 PID 映射到 0..num_classes-1
        :param max_samples: 只读取前 N 张图，用于快速测试流程
        :param max_pids: 只读取前 N 个身份，用于快速测试流程
        :param max_images_per_pid: 每个身份最多读取 N 张图，用于均衡快速测试
        """
        assert mode in ['train', 'query', 'gallery']
        self.mode = mode
        self.transform = transform
        self.root_dir = Path(root_dir)
        self.data = []
        self.pid2label = {}
        self.label2pid = {}

        if mode == 'train':
            dir_path = self.root_dir / 'bounding_box_train'
        elif mode == 'query':
            dir_path = self.root_dir / 'query'
        else:
            dir_path = self.root_dir / 'bounding_box_test'

        if not dir_path.exists():
            raise FileNotFoundError(
                f"找不到 Market1501 子目录: {dir_path}. "
                "请确认 root_dir 下包含 bounding_box_train/query/bounding_box_test。"
            )

        img_list = sorted(f for f in os.listdir(dir_path) if f.lower().endswith('.jpg'))

        for img_name in img_list:
            # 文件名格式: 1501_c1s1_000151_03.jpg
            pid, cam = self._parse_filename(img_name)
            if pid <= 0:
                continue
            img_path = dir_path / img_name
            self.data.append((img_path, pid, cam))

        if max_pids is not None or max_images_per_pid is not None:
            selected = []
            pid_counts = {}
            allowed_pids = sorted({pid for _, pid, _ in self.data})
            if max_pids is not None:
                allowed_pids = allowed_pids[:max_pids]
            allowed_pids = set(allowed_pids)

            for item in self.data:
                _, pid, _ = item
                if pid not in allowed_pids:
                    continue
                count = pid_counts.get(pid, 0)
                if max_images_per_pid is not None and count >= max_images_per_pid:
                    continue
                selected.append(item)
                pid_counts[pid] = count + 1
            self.data = selected

        if max_samples is not None:
            self.data = self.data[:max_samples]

        if relabel:
            pids = sorted({pid for _, pid, _ in self.data})
            self.pid2label = {pid: label for label, pid in enumerate(pids)}
            self.label2pid = {label: pid for pid, label in self.pid2label.items()}
            self.data = [(path, self.pid2label[pid], cam) for path, pid, cam in self.data]

        if not self.data:
            raise RuntimeError(f"{dir_path} 中没有可用的 .jpg 图像。")

    def _parse_filename(self, filename):
        # 1501_c1s1_000151_03.jpg → pid=1501, camid=1
        splits = filename.split('_')
        pid = int(splits[0])
        camid = int(splits[1][1])  # e.g., c1 → 1
        return pid, camid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, pid, camid
