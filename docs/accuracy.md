## `evaluate_accuracy`

## **Description**
We can call `evaluate_accuracy` to see how many tokens a trained model correctly predicts on the validation set.

## **Usage**
```python
from src.analysis import accuracy

percent_correct = accuracy(
    checkpoint_path="./checkpoints/classical_example/model_epoch_3.pt",
    evaluation_batch_size=32,
    choose_best_val_epoch=False,
    device="gpu",
    qpu_count=-1,
)
```