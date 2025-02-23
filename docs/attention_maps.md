# `get_attention_maps`

## **Description**
To visualize the attention map of a SMILES string, we can use the `get_attention_maps` function. We provide the function with the trained checkpoint file, the directory where the figures will be saved to, and the list of SMILES string to generate an attention map with.

## **Parameters**
- checkpoint_path (str): Path to the model checkpoint.
- save_dir (str): Directory to save the attention maps.
- smiles_list (Union[str, List[str]]): A single SMILES string or a list of SMILES strings.
- choose_best_val_epoch (bool, default=True): Choose the best validation epoch for evaluation from any - previous epoch.
- show_plots (bool, default=False): Whether to display the attention map plots.
- device (str, default="gpu"): Device for computation ("cpu" or "gpu").
- qpu_count (int, default=-1): Number of GPUs to use (-1 = all available GPUs).

## **Returns**
- None: The function generates and saves attention maps but does not return a value.

## **Usage**
```python
from src.analysis import get_attention_maps

get_attention_maps(
    checkpoint_path="./checkpoints/classical_example/model_epoch_3.pt",
    save_dir="./attention_maps/classical_example/",
    smiles_list=["CCN(CC)CC"],
    choose_best_val_epoch=True,
    show_plots=True,
    device="gpu",
    qpu_count=-1,
)
```

Multiple SMILES strings may be listed in `smiles_list` and each attention map will be saved in `save_dir`.