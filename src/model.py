# src/model.py
import torch
import torch.nn as nn

class CapnoCNN(nn.Module):
    def __init__(self):
        super(CapnoCNN, self).__init__()
        # Input shape: [Batch, 1, 3000] (Because CapnoBase is 300Hz)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7), # 3000 -> 1500
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # 1500 -> 750
            
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), # 750 -> 375
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2) # 375 -> 187
        )
        
        self.classifier = nn.Sequential(
            # The flattened size is now 32 channels * 187 length = 5984
            nn.Linear(32 * 187, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x