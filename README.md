# PSLG Model Construction for GO Prediction

This repository facilitates the training and evaluation of the GenDis model for Gene Ontology (GO) prediction tasks. The provided training script `construct_gendis.py` is highly configurable and supports tasks including **training**, **testing**, and **model fine-tuning**.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Configuration](#configuration)
- [Options](#options)
- [Details](#details)
- [License](#license)

---

## Requirements

Before running the script, ensure you have the following installed:

1. **Python 3.10+**
2. Required Python libraries (install via `requirements.txt` if provided in your repository or modify the command below as needed):
   ```bash
   pip install -r requirements.txt
   ```

**Hardware Requirements:**
- GPU support is strongly recommended for deep learning tasks. Ensure CUDA is properly set up and GPU IDs are provided correctly when executing the script.

---

## Usage

The primary script for this repository is `construct_gendis.py`. It is used for training and evaluating protein embeddings on GO tasks. The general command structure is as follows:

```shell
DATADIR=/path/to/data
DATE=$(date +%d-%m)  # Example of creating date dynamically
ONTOLOGY=(cc mf bp)
ONT=${ONTOLOGY[2]}      # Use 'cc', 'mf', or 'bp'

python scripts/construct_gendis.py -c configs/training_netgo-v1/${ONT}o-${DATE}.yml \
    $DATADIR/dataset_state_dict.pkl \
    $DATADIR/MSAs/ \
    $DATADIR/training/trained_model/model-${DATE}/${ONT}o/
```

### Example Commands

### Training stage 1
```bash
python scripts/construct_gendis.py -c configs/training_msa-v1/bpo-7-26.yml \
    /path/to/dataset_state_dict.pkl \
    /path/to/MSAs/ \
    /path/to/save/model/
```

### Training stage 2
```bash
python scripts/construct_gendis.py -c configs/training_msa-v1/bpo-8-24.yml \
    /path/to/dataset_state_dict.pkl \
    /path/to/MSAs/ \
    /path/to/save/model/
```

### Testing
To evaluate a trained model:
```bash
python scripts/construct_gendis.py -c configs/evaluating_msa-v1/bpo-8-24.yml \
    /path/to/dataset_state_dict.pkl \
    /path/to/MSAs/ \
    /path/to/trained/model/
```

### Evaluating with Average Testing

The `average_performance.py` script provides an evaluation method based on averaging over multiple sampling steps. This process enhances the accuracy of performance metrics such as **Fmax** and **Area Under Precision-Recall Curve (AuPRC)** by reducing noise due to stochasticity.

#### Command Example:
```bash
python scripts/average_performance.py \
    -c configs/testing_config.yml \
    -n 5 \
    --load trained_model.pth \
    -ps /path/to/save/evaluations.npy \
    /path/to/dataset_state_dict.pkl \
    /path/to/MSAs/ \
    /path/to/trained/models/
```

#### Process:
1. The script computes predictions for a given test dataset.
2. It averages predictions over multiple sampling steps (controlled by `-n <num-samplings>`).
3. Evaluates metrics such as:
   - **Fmax Score**: Measures the maximum F-score across thresholds.
   - **AuPRC**: Provides an area-based metric for precision-recall performance.

#### Key Arguments:
- `-c` or `--config`: Configuration file path (optional if all arguments are explicitly provided).
- `-n` or `--num-samplings`: Number of sampling steps to perform averaging (default: `5`).
- `--load`: Path to a pre-trained model file.
- `-ps` or `--prediction-save`: Path to save the evaluation results in `.npy` format.

#### Output:
The script will output metrics like **Fmax** and **AuPRC** in the terminal and optionally save the prediction results in the specified path.

---

## Configuration

The script allows for specifying parameters using a configuration `.yml` file. It is highly recommended to predefine key training parameters, such as epochs, learning rate, and batch size, inside this file. Pass the configuration file using the `-c` or `--config` argument.

### Sample Configuration File
```yaml
# Example: configs/training_netgo-v1/bp.yml
mode: train
task: biological_process
epochs: 100
batch_size: 32
lr: 0.0001
top_k: 40
max_len: 2000
# Many other options are supported!
```

---

## Options

The script supports a wide variety of command-line arguments. Below is an overview of the most commonly used ones:

### General Arguments
- `file_address`: Path to the dataset file.
- `working_dir`: Directory for multiple sequence alignment (MSA) files.
- `model_saving`: Directory to save trained model.

### Model Configuration
- `--netG`: Specify the encoder network architecture (`resnet_9blocks`, `resnet_6blocks`, etc.).
- `--ngf`: Number of generator filters in the last convolutional layer.
- `--normG`: Normalization type for the generator (`instance`, `batch`, `none`).

### Training Parameters
- `--mode`: Mode of operation - `train`, `test`, etc.
- `--batch-size`: Batch size (default: `32`).
- `--epochs`: Number of epochs for training.
- `--lr`: Learning rate.

### Data Handling
- `--top-k`: Number of top sequences used from MSAs.
- `--max-len`: Maximum sequence length to consider.
- `--msa-encoding-strategy`: Encoding method for MSA files (`one_hot`, `emb_plus_one_hot`, `fast_dca`, etc.).

### Hardware
- `--gpu-ids`: GPU IDs to use (e.g., `'0,1'` for multi-GPU training, or `-1` for CPU).
- `--amp`: Use automatic mixed precision (for faster training on GPUs).

### Evaluation-Specific Parameters
- **In `average_performance.py`**:
  - `-n` or `--num-samplings`: Number of sampling passes for average evaluation.
  - `-ps` or `--prediction-save`: Path to save predictions.
  - Outputs **Fmax** and **AuPRC** metrics.

---

## Details

The script operates in several modes based on the `--mode` argument:
- `train`: Train your GenDis model using MSA and gene ontology annotations.
- `test`: Run evaluation on a pre-trained model to generate predictions.
- `train_ipr` or `test_ipr`: Special options for training/testing with InterPro features.
- `train_im` or `test_im`: Train/test IMEncoder-based features.

### Key Functionalities
1. **Training**: Supports various model types, including resnet-based encoders and InterPro features.
2. **Evaluation**: Produce prediction results to a file or console.
3. **Pre-trained Models**: Option to load or fine-tune a pre-trained model from saved state dictionaries (`--load` or `--for-retrain`).

---

## License

This project is distributed under the MIT License. See `LICENSE.md` for more details.

---