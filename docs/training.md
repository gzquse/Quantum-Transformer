# `train_transformer`

## **Description**
The `train_transformer` function trains a transformer model with either classical or quantum attention mechanisms. It supports various hyperparameters to customize training, including learning rate, batch size, number of epochs, and quantum-specific settings such as the number of qubits and quantum gradient calculation.

During the training, the parameters for each epoch are saved in the specified checkpoint_dir, along with `most_recent_batch.pt` if the `save_every_n_batches` argument is specified.

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