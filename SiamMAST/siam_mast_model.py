import torch
import torch.nn as nn
from .cnn import CNN
from .lstm import LSTM
from .motion_aware import SpatialMotionAwareness, TemporalMotionAwareness

class SiamMAST(nn.Module):
    def __init__(self, num_classes=101):
        super(SiamMAST, self).__init__()
        self.cnn1 = CNN()
        self.cnn2 = CNN()
        self.lstm1 = LSTM()
        self.lstm2 = LSTM()
        self.spatial_motion_aware = SpatialMotionAwareness()
        self.temporal_motion_aware = TemporalMotionAwareness()
        self.fc = nn.Linear(4096 + 256, num_classes)  # 4096 from CNN, 256 from LSTM

    def forward(self, input1, input2):
        feat1 = self.cnn1(input1)
        feat2 = self.cnn2(input2)

        temp_feat1 = self.lstm1(feat1)
        temp_feat2 = self.lstm2(feat2)

        spatial_feat = self.spatial_motion_aware(feat1, feat2)
        temporal_feat = self.temporal_motion_aware(temp_feat1, temp_feat2)

        combined_feat = torch.cat((spatial_feat, temporal_feat), dim=1)
        output = self.fc(combined_feat)
        return output