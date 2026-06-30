import matplotlib.pyplot as plt
import torchvision.transforms as T

from reid_baseline.market1501_loader import Market1501Dataset

# 指定你的 Market1501 数据集路径
data_root = '/path/to/Market-1501-v15.09.15'  # 替换为你的数据集路径

transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor()
])

dataset = Market1501Dataset(root_dir=data_root, mode='train', transform=transform)

# 展示前两张图像及其标签
for i in range(2):
    img, pid, camid = dataset[i]
    img_np = img.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    plt.title(f'Person ID: {pid}, Camera ID: {camid}')
    plt.axis('off')
    plt.show()
