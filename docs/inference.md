# `generate_smiles`

## **Description**
We can call the `generate_smiles` function to perform inference on our trained model. The output is the validity, uniqueness, and novelty percentages for the generated sequences. Valid sequences are those where the resulting string can be successfully parsed by RDKit. Novel SMILES strings are those which do not appear in the training set.

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

