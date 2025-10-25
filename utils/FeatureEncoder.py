from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

@dataclass
class FeatureSchema:
    '''
    name: Name of the feature
    type: Type of the feature (categorical, numerical, embedding)
    description: Description of the feature
    input_dim: Input dimension of the feature (no of unique values/classes for categorical features)
    output_dim: Output dimension of the feature
    '''
    name: str
    type: Literal['categorical', 'numerical','embedding']
    description: str
    input_dim: int
    output_dim: int

class FeatureEncoder(nn.Module):
    def __init__(self, schema: FeatureSchema):
        """
        Initializes the feature encoder.

        Args:
            schema (FeatureSchema): The schema of the features to be encoded.

        The encoder will be initialized with the appropriate modules based on the feature types in the schema.
        """
        super().__init__()
        self.schema=schema
        self.encoders=nn.ModuleDict()

        for feature in schema:
            if feature.type == 'categorical':
                self.encoders[feature.name] = nn.Embedding(feature.input_dim, feature.output_dim)
            elif feature.type == 'numerical':
                if feature.input_dim == 1:
                    self.encoders[feature.name] = nn.Identity()
                else:
                    self.encoders[feature.name] = nn.Linear(feature.input_dim, feature.output_dim)
            elif feature.type == 'embedding':
                self.encoders[feature.name] = nn.Identity()
            else:
                raise ValueError(f"Unknown feature type: {feature.type}")
    def forward(self, x):
        out=[]
        for name,encoder in self.encoders.items():
            out.append(encoder(x[name]))
        return torch.cat(out, dim=1)
    



