import torch
import torch.nn as nn

class FeatureAdapter(nn.Module):
    """
    Lightweight MLP adapter for medical feature refinement.
    Architecture: Linear(512 -> 1024) -> ReLU -> Dropout -> Linear(1024 -> 512) -> LayerNorm
    Includes a Residual connection.
    """
    def __init__(self, input_dim=512, hidden_dim=1024, dropout=0.1):
        super(FeatureAdapter, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        return self.layernorm(x + residual)

class DiseaseClassifier(nn.Module):
    """
    Downstream classification head for specific medical conditions.
    """
    def __init__(self, input_dim=512, num_classes=2):
        super(DiseaseClassifier, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.head(x)
