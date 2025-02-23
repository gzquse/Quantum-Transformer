# `generate_smiles`

## **Description**
We can call the `generate_smiles` function to perform inference on our trained model. The output is the validity, uniqueness, and novelty percentages for the generated sequences. Valid sequences are those where the resulting string can be successfully parsed by RDKit. Novel SMILES strings are those which do not appear in the training set.

## **Parameters**
- checkpoint_path (str): Path to the model checkpoint.
- save_dir (str): Path to save the generated SMILES strings.
- choose_best_val_epoch (bool, default=True): Choose the best validation epoch for evaluation from any - previous epoch.
- num_of_model_queries (int, default=100000): Number of attempts to generate SMILES strings.
- sampling_batch_size (int, default=10000): Batch size for sampling.
- MW (Union[float, np.float64], default=np.nan): Molecular weight for conditional sampling.
- HBA (Union[float, np.float64], default=np.nan): Hydrogen bond acceptors for conditional sampling.
- HBD (Union[float, np.float64], default=np.nan): Hydrogen bond donors for conditional sampling.
- nRot (Union[float, np.float64], default=np.nan): Number of rotatable bonds for conditional sampling.
- nRing (Union[float, np.float64], default=np.nan): Number of rings for conditional sampling.
- nHet (Union[float, np.float64], default=np.nan): Number of heteroatoms for conditional sampling.
- TPSA (Union[float, np.float64], default=np.nan): Topological polar surface area for conditional sampling.
- LogP (Union[float, np.float64], default=np.nan): LogP for conditional sampling.
- StereoCenters (Union[float, np.float64], default=np.nan): Number of stereocenters for conditional sampling.
- imputation_method (str, default="knn"): Imputation method for missing physicochemical properties.
- imputation_dataset_path (Optional[str], default=None): Path to the imputation dataset. Defaults to the - training dataset specified by the train_id from the checkpoint.
- dataset_novelty_check_path (Optional[str], default=None): Path to the dataset for novelty check. Defaults - to the training dataset specified by the train_id from the checkpoint.
- device (Union[str, torch.device], default="gpu"): Device for sampling ("cpu" or "gpu").
- qpu_count (int, default=-1): Number of GPUs to use (-1 = all available GPUs).

## **Returns**
Tuple[float, float, float]: A tuple containing:

- Average validity of the generated SMILES strings.
- Average uniqueness of the generated SMILES strings.
- Average novelty of the generated SMILES strings.

## **Usage**
```python
from src.analysis import generate_smiles

valid, unique, novel = generate_smiles(
    checkpoint_path="./checkpoints/classical_example/model_epoch_3.pt",
    save_dir="./generated_molecules/classical_example.csv",
    choose_best_val_epoch=True,
    num_of_model_queries=1000,
    sampling_batch_size=250,
    imputation_dataset_path="./dataset/train_dataset.csv",
    dataset_novelty_check_path="./dataset/train_dataset.csv",
    device="gpu",
)
```

If a model was trained with molecular property embeddings, the `generate_smiles` function allows it to be sampled from conditionally and grants us the option to specify molecular properties. The following properties can be specified:

- MW (molecular weight)
- HBA (number of hydrogen bond acceptors) 
- HBD (number of hydrogen bond donors) 
- nRot (number of rotatable bonds) 
- nRing (number of rings)
- nHet (number of heteroatoms)
- TPSA (topological polar surface area)
- LogP (octanol-water partition coefficent)
- StereoCenters (number of stereocenter)

Any properies not specified are imputed from the training data. The imputation is performed with Scikit-learn using KNN by default, but can be specified using the `imputation_method` argument with the options of:
`mean`, `median`, `most_frequent`, `multivariate`, or `knn`. 

Below is an example of two sampling experiments where molecules are sampled to have a target weight of 120:

```python
from src.analysis import generate_smiles

generate_smiles(
    checkpoint_path="./checkpoints/classical_example_conditions/model_epoch_3.pt",
    save_dir="./generated_molecules/classical_example_conditions_MW_120.csv",
    num_of_model_queries=1000,
    sampling_batch_size=250,
    imputation_dataset_path="./dataset/train_dataset.csv",
    dataset_novelty_check_path="./dataset/train_dataset.csv",
    device="gpu",
    MW=120,
)
```

