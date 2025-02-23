# `train_transformer`

## **Description**
The `train_transformer` function trains a transformer model with either classical or quantum attention mechanisms. It supports various hyperparameters to customize training, including learning rate, batch size, number of epochs, and quantum-specific settings such as the number of qubits and quantum gradient calculation.

## **Parameters**
- training_data (str): Path to the training data CSV file.
- checkpoint_dir (str): Directory to save model checkpoints.
- checkpoint_resume_path (Optional[str], default=None): Path to resume training from a checkpoint.
- learning_rate (float, default=0.005): Learning rate for the optimizer.
- weight_decay (float, default=0.1): Weight decay for the optimizer.
- batch_size (int, default=256): Batch size for training.
- epochs (int, default=20): Number of epochs to train.
- save_every_n_batches (int, default=0): Frequency to save batch-level model checkpoints (0 means no batch-level saving).
- validation_split (float, default=0.05): Fraction of data used for validation.
- attn_type (str, default="classical"): Type of attention mechanism ("classical" or "quantum").
- num_qubits (int, default=6): Number of working qubits in the quantum attention layer.
- ansatz_layers (int, default=1): Number of layers in the quantum ansatz.
- conditional_training (bool, default=True): Whether to train with physicochemical properties.
- quantum_gradient_method (str, default="spsa"): Quantum gradient method ("spsa" or "parameter-shift").
- spsa_epsilon (float, default=0.01): Epsilon value for SPSA optimization.
- sample_percentage (float, default=1.0): Fraction of dataset used for training.
- seed (int, default=42): Random seed for reproducibility.
- classical_parameter_reduction (bool, default=False): Ensure the number of classical parameters matches the number of quantum parameters.
- device (str, default="gpu"): Device for training ("cpu" or "gpu").
- qpu_count (int, default=-1): Number of GPUs to use (-1 = all available GPUs).

## **Returns**
- None: The function trains the model and saves checkpoints as specified but does not return a value.
## **Usage**
The below example runs a fully classical model where the number of parameters is equal to the number of parameters used in the quantum model. This is triggered by setting the `classical_parameter_reduction` argument to `True`.

```python
from src.train import train_transformer

train_transformer(
    training_data="./dataset/qm9.csv",
    checkpoint_dir="./checkpoints/classical_example/",
    checkpoint_resume_path=None,
    learning_rate=0.005,
    weight_decay=0.1,
    batch_size=256,
    epochs=3,
    save_every_n_batches=0,
    validation_split=0.05,
    attn_type="classical",
    num_qubits=6,
    ansatz_layers=1,
    conditional_training=False,
    quantum_gradient_method="spsa",
    spsa_epsilon=0.01,
    sample_percentage=1.0,
    seed=42,
    classical_parameter_reduction=True,
    device="gpu",
    qpu_count=-1,
)
```
---
To train the model with molecular property embeddings, `conditional_training` gets set to `True`:
```python
from src.train import train_transformer

train_transformer(
    training_data="./dataset/qm9.csv",
    checkpoint_dir="./checkpoints/classical_example_conditions/",
    checkpoint_resume_path=None,
    learning_rate=0.005,
    weight_decay=0.1,
    batch_size=256,
    epochs=3,
    save_every_n_batches=0,
    validation_split=0.05,
    attn_type="classical",
    num_qubits=6,
    ansatz_layers=1,
    conditional_training=True,
    quantum_gradient_method="spsa",
    spsa_epsilon=0.01,
    sample_percentage=1.0,
    seed=42,
    classical_parameter_reduction=True,
    device="gpu",
    qpu_count=-1,
)
```
---
The quantum model can be run by switching the `attn_type` argument to `'quantum'`:
```python
from src.train import train_transformer

train_transformer(
    training_data="./dataset/qm9.csv",
    checkpoint_dir="./checkpoints/quantum_example/",
    checkpoint_resume_path=None,
    learning_rate=0.005,
    weight_decay=0.1,
    batch_size=256,
    epochs=3,
    save_every_n_batches=0,
    validation_split=0.05,
    attn_type="quantum",
    num_qubits=6,
    ansatz_layers=1,
    conditional_training=False,
    quantum_gradient_method="spsa",
    spsa_epsilon=0.01,
    sample_percentage=1.0,
    seed=42,
    classical_parameter_reduction=True,
    device="gpu",
    qpu_count=-1,
)
```