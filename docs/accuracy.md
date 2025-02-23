## `evaluate_accuracy`

## **Description**
We can call `evaluate_accuracy` to see how many tokens a trained model correctly predicts on the validation set.

## **Parameters**
- checkpoint_path (str): Path to the model checkpoint.
- evaluation_batch_size (int, default=32): Batch size for evaluation.
- choose_best_val_epoch (bool, default=False): Choose the best validation epoch for evaluation from any - previous epoch.
- device (str, default="gpu"): Device for evaluation ("cpu" or "gpu").
- qpu_count (int, default=-1): Number of GPUs to use (-1 = all available GPUs).

## **Returns**
- float: Next-token prediction accuracy on the validation dataset.

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