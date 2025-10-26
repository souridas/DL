# Deep Learning Utilities

This repository is dedicated to building utilities that facilitate the creation of scalable, robust, and reliable Deep Learning Systems.

## Utilities

### `utils/FeatureEncoder.py`

This module provides two key utilities:

- **`FeatureSchema`**: Helps in maintaining feature metadata such as name, description, type, and dimensions.
- **`FeatureEncoder`**: Designed to take a `FeatureSchema` as input.

#### Benefits of `FeatureEncoder`

- **Reliable Data**: Ensures data consistency for the model, as features are stored as a dictionary, eliminating concerns about input order.
- **Robustness**: Easily accommodates new features by simply adding a new item to the feature schema. This also allows leveraging previous checkpoint weights for iterative training.

#### Testing

You can test the `utils/FeatureEncoder.py` utility by running `examples/test1.py`.

### `utils/ContinualTraining.py`

This module provides utilities for continual training of deep learning models when new FeatureSchema changes or Model architecture changes.

#### Testing

You can test the `utils/ContinualTraining.py` utility by running `examples/test2.py`.
