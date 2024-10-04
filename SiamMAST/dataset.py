import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, transform=None):
    # Initialize data set, load videos and tags
        self.transform = transform
        self.data = [] 

    def __len__(self):
        return len(self.data)  

    def __getitem__(self, idx):
        video1 = self.data[idx]['video1']
        video2 = self.data[idx]['video2']
        label = self.data[idx]['label']

        if self.transform:
            video1 = self.transform(video1)
            video2 = self.transform(video2)

        return video1, video2, label
