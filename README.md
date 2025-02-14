# A Hybrid Transformer Architecture with a Quantized Self-Attention Mechanism Applied to Molecular Generation.
This is a repository containing code for a hybrid quantum-classical transformer model from the paper: A Hybrid Transformer Architecture with a Quantized Self-Attention Mechanism Applied to Molecular Generation.

## Installation

To reproduce the results from this repository, you can set up your environment using **Conda** or **pip**.

First, clone the repository:
```sh
git clone https://github.com/anthonysmaldone/Quantum-Transformer.git
cd Quantum-Transformer
```
### Using Conda (recommended for Perlmutter jobs):
```sh
conda env create -f cuda_quantum_transformer_env.yml
conda activate cuda_quantum_transformer_env
```

### Using Pip
```sh
pip install cudaq==0.9.1 rdkit==2024.9.4 torch==2.5.1+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    pandas==2.2.2 torchdata==0.10.1 tqdm==4.67.1 \
    scikit-learn==1.5.1 seaborn==0.13.2 gdown==5.2.0
```
## Reproducing Results

This repository provides a script, `reproduce.py`, to facilitate the reproduction of key results from the paper, including model training, figure generation, and inference.

The trained checkpoint files for the best and last (20th) epoch for each models are located in `model_checkpoints/`. The figures and inference reproducability functions uses these models to recreate the data from the paper.

### **1. Generate Figures**
To generate figures used in the paper, use the --figures flag.
```sh
python reproduce.py --figures
```
No additional options are required.

### **2. Run Inference**
To perform inference on a trained model, use the `--inference-results` flag along with the required `--model` and `--mode options`.
```sh
python reproduce.py --inference-results --model <MODEL_TYPE> --mode <MODE>
```
Model Options (--model)
- quantum – Train the quantum model.
- classical_eq – Train the classical equivalent model.
- classical – Train the standard classical model.
  
Training Mode Options (--mode)
- sequence – Train using sequence-only data.
- conditions – Train using both sequence and condition data.

Examples
- Run inference on the pre-trained quantum model with SMILES data only:
```sh
python reproduce.py --inference-results --model classical --mode sequence
```
- Run inference on the pre-trained classical model with SMILES and physicochemical properties:
```sh
python reproduce.py --inference-results --model quantum --mode conditions
```

### **3. Train Models**
To re-train a model, use the `--train-models` flag along with the required `--model` and `--mode` options.

```sh
python reproduce.py --train-models --model <MODEL_TYPE> --mode <MODE>
```
Model Options (--model)
- quantum – Train the quantum model.
- classical_eq – Train the classical equivalent model.
- classical – Train the standard classical model.
  
Training Mode Options (--mode)
- sequence – Train using sequence-only data.
- conditions – Train using both sequence and condition data.

Examples
- Train the quantum model with SMILES data only:
```sh
python reproduce.py --train-models --model quantum --mode sequence
```
- Train the classical model with SMILES and physicochemical properties:
```sh
python reproduce.py --train-models --model classical_eq --mode conditions
```

**Note**: Exact numerical reproducability is not garunteed across hardware architectures. These models were trained on Perlmutter using 4 NVIDIA A100 GPUs.

## Usage
A notebook tutorial can be found here (LINK) and documentation on available functions can be found in `docs/`.