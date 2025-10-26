from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Path setup for local imports ---
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from utils.FeatureEncoder import FeatureEncoder, FeatureSchema
from utils.ContinualTraining import TrainingState


# ------------------------------
# 1️⃣ Define base schema
# ------------------------------
schema_v1 = [
    FeatureSchema(name="age", type="numerical", description="Age of user", input_dim=1, output_dim=1),
    FeatureSchema(name="income", type="numerical", description="Annual income", input_dim=1, output_dim=1),
    FeatureSchema(name="gender", type="categorical", description="Gender of user", input_dim=2, output_dim=4),
    FeatureSchema(name="city", type="categorical", description="City ID", input_dim=100, output_dim=8),
    FeatureSchema(name="profile_vec", type="embedding", description="Precomputed embedding vector", input_dim=128, output_dim=128),
]


# ------------------------------
# 2️⃣ Create base model
# ------------------------------
class Model(nn.Module):
    def __init__(self, schema: list[FeatureSchema], hidden_dim: int = 64):
        super().__init__()
        self.encoder = FeatureEncoder(schema)
        output_dim = sum(f.output_dim for f in schema)
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


# Initialize base model
model_v1 = Model(schema_v1)
print("Base Model:")
print(model_v1.state_dict())

# Example input batch
x_dict = {
    "age": torch.tensor([[25.0], [32.0]]),
    "income": torch.tensor([[75000.0], [90000.0]]),
    "gender": torch.tensor([0, 1]),
    "city": torch.tensor([10, 3]),
    "profile_vec": torch.randn(2, 128)
}

y = model_v1(x_dict)
#print("Output :", y)


# ------------------------------
# 3️⃣ Save checkpoint
# ------------------------------
checkpoint = TrainingState(
    checkpoint_schema=schema_v1,
    checkpoint_model_state=model_v1.state_dict()
)


# ------------------------------
# 4️⃣ Define new schema (added new feature)
# ------------------------------
schema_v2 = schema_v1 + [
    FeatureSchema(
        name="credit_score",
        type="numerical",
        description="Credit score of user",
        input_dim=1,
        output_dim=1
    )
]


# ------------------------------
# 5️⃣ Define new model version (architecture change)
# ------------------------------
class ModelV2(nn.Module):
    def __init__(self, schema: list[FeatureSchema], hidden_dim: int = 128):  # hidden_dim changed!
        super().__init__()
        self.encoder = FeatureEncoder(schema)
        output_dim = sum(f.output_dim for f in schema)
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_extra = nn.Linear(hidden_dim, hidden_dim // 2)  # New layer added
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_extra(x))  # New intermediate layer
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# ------------------------------
# 6️⃣ Load model using TrainingState
# ------------------------------
checkpoint.new_schema = schema_v2
checkpoint.new_model_class = ModelV2

model_v2 = checkpoint.load_model()

print("\n✅ Loaded Model v2:")
print(model_v2.state_dict())

x_dict_2 = {
    "age": torch.tensor([[25.0], [32.0]]),
    "income": torch.tensor([[75000.0], [90000.0]]),
    "gender": torch.tensor([0, 1]),
    "city": torch.tensor([10, 3]),
    "profile_vec": torch.randn(2, 128),
    "credit_score": torch.tensor([[700.0], [800.0]])
}

y = model_v2(x_dict_2)
#print("Output :", y)
