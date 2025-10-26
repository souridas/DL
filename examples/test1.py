from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from utils.FeatureEncoder import FeatureEncoder, FeatureSchema
import torch
import torch.nn as nn
import torch.nn.functional as F


schema = [
    FeatureSchema(name="age", type="numerical", description="Age of user", input_dim=1, output_dim=1),
    FeatureSchema(name="income", type="numerical", description="Annual income", input_dim=1, output_dim=1),
    FeatureSchema(name="gender", type="categorical", description="Gender of user", input_dim=2, output_dim=4),
    FeatureSchema(name="city", type="categorical", description="City ID", input_dim=100, output_dim=8),
    FeatureSchema(name="profile_vec", type="embedding", description="Precomputed embedding vector", input_dim=128, output_dim=128),
]

encoder = FeatureEncoder(schema)
print(encoder)

# ------------------------------
# example data
# ------------------------------
x_dict = {
    "age": torch.tensor([[25.0], [32.0]]),                # (batch, 1)
    "income": torch.tensor([[75000.0], [90000.0]]),       # (batch, 1)
    "gender": torch.tensor([0, 1]),                       # (batch,)
    "city": torch.tensor([10, 3]),                        # (batch,)
    "profile_vec": torch.randn(2, 128)                    # (batch, 128)
}

encoded = encoder(x_dict)
print("Encoded shape:", encoded.shape)

# ------------------------------
# model
# ------------------------------    
class Model(nn.Module):
    def __init__(self,schema: list[FeatureSchema],hidden_dim: int=64):
        super().__init__()
        self.encoder = FeatureEncoder(schema)
        output_dim = sum(feature.output_dim for feature in schema)
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

model = Model(schema)
print(model)

y = model(x_dict)
print(y)
print("Output shape:", y.shape)