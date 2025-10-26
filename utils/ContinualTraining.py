from dataclasses import dataclass
from typing import Literal, List
from utils.FeatureEncoder import FeatureSchema

@dataclass
class TrainingState:
    checkpoint_schema: List[FeatureSchema]
    checkpoint_model_state: dict
    new_schema: List[FeatureSchema] = None  
    new_model_class: dict = None


    def load_model(self):
        if self.new_model_class is None:
            raise ValueError("`Model_class` must be provided to load model.") 
        schema= self.new_schema or self.checkpoint_schema
        model = self.new_model_class(schema)
        model_state = model.state_dict()
        weights={
            k: v for k, v in self.checkpoint_model_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model_state.update(weights)
        model.load_state_dict(model_state)
        return model


        

        