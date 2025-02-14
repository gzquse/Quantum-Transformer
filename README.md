# Quantum-Transformer
This is a repository containing code for a hybrid quantum-classical transformer model from the paper: A Hybrid Transformer Architecture with a Quantized Self-Attention Mechanism Applied to Molecular Generation.

## Reproducing Results

This repository provides a script, `reproduce.py`, to facilitate the reproduction of key results from the paper, including model training, figure generation, and inference.

### Usage

To use the script, run the following commands from the project root:

#### **1. Generate Figures**
To generate figures used in the paper, use the --figures flag.
```sh
python reproduce.py --figures
```
No additional options are required.

#### **2. Run Inference**
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

#### **3. Train Models**
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

Note that exact numerical reproducability is not garunteed across hardware architectures. These models were trained on Perlmutter using 4 NVIDIA A100 GPUs.