# data/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        # Initialize the dataset with the video directory and transformation
        self.video_dir = video_dir
        self.transform = transform
        self.video_files = os.listdir(video_dir)

    def __len__(self):
        # Return the total number of videos
        return len(self.video_files)

    def __getitem__(self, idx):
        # Get the video frames and corresponding label
        video_path = os.path.join(self.video_dir, self.video_files[idx])
        label = self.get_label_from_filename(self.video_files[idx])  # Extract label from filename

        frames = self.load_video(video_path)  # Load video frames
        if self.transform:
            frames = self.transform(frames)  # Apply transformations if any

        return frames, label

    def load_video(self, video_path):
        # Load the video frames
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def get_label_from_filename(self, filename):
        # Extract the label from the video filename
        return int(filename.split('_')[1])  # Adjust according to your filename structure
