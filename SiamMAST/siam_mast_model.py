import torch
import torch.nn as nn
from .cnn import CNN
from .lstm import LSTM

# model/siam_mast.py


class SiamMAST(nn.Module):
    def __init__(self, num_classes):
        super(SiamMAST, self).__init__()

        # Initialize CNNs and LSTMs for the Siamese architecture
        self.cnn1 = self._build_cnn()
        self.cnn2 = self._build_cnn()
        self.lstm1 = nn.LSTM(input_size=4096, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=4096, hidden_size=256, batch_first=True)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _build_cnn(self):
        # Build a CNN model (e.g., AlexNet)
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Add additional layers here as needed
        )
        return model

    def forward(self, input1, input2):
        # Forward pass for both CNNs
        feature1 = self.cnn1(input1)
        feature2 = self.cnn2(input2)

        # Flatten the features and pass through LSTM
        feature1 = feature1.view(feature1.size(0), -1, 4096)
        feature2 = feature2.view(feature2.size(0), -1, 4096)

        lstm_out1, _ = self.lstm1(feature1)
        lstm_out2, _ = self.lstm2(feature2)

        # Concatenate the outputs for classification
        combined_features = torch.cat((lstm_out1[:, -1, :], lstm_out2[:, -1, :]), dim=1)
        x = self.fc1(combined_features)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
