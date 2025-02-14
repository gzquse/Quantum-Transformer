# `get_attention_maps`

## **Description**
To visualize the attention map of a SMILES string, we can use the `get_attention_maps` function. We provide the function with the trained checkpoint file, the directory where the figures will be saved to, and the list of SMILES string to generate an attention map with.

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