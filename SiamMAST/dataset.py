import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, transform=None):
        # 初始化数据集，加载视频和标签
        self.transform = transform
        self.data = []  # 在此处加载数据，例如视频路径和标签

    def __len__(self):
        return len(self.data)  # 返回数据集大小

    def __getitem__(self, idx):
        # 加载视频帧和标签
        video1 = self.data[idx]['video1']
        video2 = self.data[idx]['video2']
        label = self.data[idx]['label']

        if self.transform:
            video1 = self.transform(video1)
            video2 = self.transform(video2)

        return video1, video2, label
