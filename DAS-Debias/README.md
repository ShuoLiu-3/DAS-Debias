# DAS-Debias

An **Activation Intervention** based framework for debiasing large language models. It precisely locates and neutralizes bias representations in the intermediate layers of LLMs through probe detection + adapter distillation + adaptive gating, while preserving the model's original capabilities as much as possible.

## Project Structure

```
DAS-Debias/
├── src/
│   ├── main.py                      # Main entry: full pipeline
│   ├── utils/
│   │   ├── model.py                 # DNN probe & NeutralizationAdapter
│   │   ├── attack.py                # PGD adversarial attack
│   │   ├── dataset.py               # Activation dataset loading
│   │   ├── globals.py               # Global constants (model paths, result dirs)
│   │   └── logger.py                # Logging utility
│   ├── evaluation/
│   │   ├── evaluate.py              # Base evaluator
│   │   ├── bbq_eval.py              # BBQ bias evaluation
│   │   ├── bias_asker.py            # BiasAsker bias evaluation
│   │   ├── mmlu.py                  # MMLU capability evaluation
│   │   └── trace.py                 # Activation tracing
│   └── runsh/
│       ├── debias_bbq.py            # BBQ batch runner
│       └── debias_biasasker.py      # BiasAsker batch runner
├── biased_knowledge/
│       ├── corpus/                      # Bias corpus (for activation tracing)
│       └── {dimension}/group.json       # Group information per dimension
├── configs/
│   └── default.yaml                     # Default hyperparameter configuration
│   
└── README.md
```

## Environment


All experiments were conducted on a workstation equipped with:
* CPU: 2 × AMD 7K62 processors
* Memory: 512 GB DDR4 2666 MHz RAM (16 × 32 GB)
* Storage: 1 TB SSD + 8 TB Seagate HDD
* GPU: 1 × NVIDIA A100 80GB


## Environment Dependencies

```bash
pip install torch transformers einops baukit scikit-learn pandas matplotlib seaborn tqdm
```

## Quick Start

### 1. Configure Model Paths

Edit `src/utils/globals.py` and update the paths in `HF_NAMES` to your local model paths:

```python
HF_NAMES = {
    'llama2_chat_7B': '/your/path/to/Llama-2-7b-chat-hf',
    'llama2_chat_13B': '/your/path/to/Llama-2-13b-chat-hf',
}
```

### 2. Run the Full Pipeline

```bash
cd src

# BBQ evaluation (using Gender_identity as an example)
python main.py \
    --model_name llama2_chat_7B \
    --trace_file ../biased_knowledge/corpus/Gender_identity_sentences.json \
    --protect_attr Gender_identity \
    --eval_task bbq \
    --eval_dataset_name BBQ \
    --test_file Gender_identity \
    --bbq_cate Gender_identity \
    --class_num 3

# BiasAsker evaluation (using Age as an example)
python main.py \
    --model_name llama2_chat_7B \
    --trace_file ../biased_knowledge/corpus/Age_sentences.json \
    --protect_attr Age \
    --eval_task biasasker \
    --eval_dataset_name BiasAsker \
    --test_file Age \
    --bbq_cate Age \
    --class_num 2

# MMLU capability verification (ensure debiasing does not affect model performance)
python main.py \
    --model_name llama2_chat_7B \
    --trace_file ../biased_knowledge/corpus/Gender_identity_sentences.json \
    --protect_attr Gender_identity \
    --eval_task mmlu \
    --eval_dataset_name MMLU \
    --test_file ../data/MMLU/mmlu.json \
    --class_num 3
```

### 3. Batch Execution

Use the scripts under `runsh/` to run across multiple bias dimensions:

```bash
cd src/runsh
python debias_bbq.py        # Batch run BBQ
python debias_biasasker.py  # Batch run BiasAsker
```

## Pipeline Details

`run_pipe()` in `main.py` executes the following 4 stages sequentially:

### Stage 1: Trace (Activation Tracing)

Uses the bias corpus under `biased_knowledge/corpus/` as input to extract the activation values of each token at each layer of the LLM, saved as `.npy` files.

### Stage 2: Probe (Probe Training)

Trains a two-layer DNN classifier (probe) for each layer to determine whether the activation values encode bias information. Selects the top `num_mlps` layers with the highest F1 scores as intervention targets (only layers with F1 > 0.4 participate in subsequent interventions).

### Stage 3: Distillation (Adapter Distillation)

For each selected layer:
1. Computes the KL divergence (risk score) for each sample using the probe
2. Performs PGD adversarial attack on high-risk samples (KL > `gating_threshold`) to generate debiased target activations
3. Trains a lightweight `NeutralizationAdapter` (bottleneck structure) to approximate the PGD perturbation

### Stage 4: Debias (Debias Inference)

During inference, for each layer's MLP output:
1. **Gating Evaluation**: Uses the probe to compute the KL divergence of the current activation
2. **Adaptive Intervention**: Only when KL > `gating_threshold`, neutralizes bias using the Adapter
3. **Steering Vector** (optional): Subtracts the bias direction vector to further eliminate residual bias

## Hyperparameter Description

For the full configuration, see [`configs/default.yaml`](configs/default.yaml). Core parameters are as follows:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `llama_7B` | Model name, must be registered in `HF_NAMES` in `globals.py` |
| `--protect_attr` | `gender` | Protected attribute dimension |
| `--class_num` | `10` | Number of bias classes (varies by dimension, see table below) |
| `--num_mlps` | `12` | Number of MLP layers selected for intervention |
| `--num_epochs` | `20` | Number of probe training epochs |
| `--noise_factor` | `5` | PGD noise intensity coefficient |
| `--num_iter` | `10` | Number of PGD iterations |
| `--eps_iter` | `4` | PGD step size = noise_factor / eps_iter |
| `--gating_threshold` | `0.2` | Default gating KL divergence threshold. Attribute-specific thresholds are configured separately in experiment scripts. |
| `--use_adapter` | `True` | Whether to use Adapter instead of PGD |
| `--distill_epochs` | `50` | Number of Adapter distillation training epochs |
| `--adapter_hidden_dim` | `512` | Adapter hidden layer dimension |
| `--consistency_alpha` | `0.1` | Consistency regularization weight (preserves original semantics) |
| `--use_steering` | `True` | Whether to enable Steering Vector |
| `--steering_alpha` | `0.02` | Steering vector strength |

### class_num Reference per Dimension

| Bias Dimension | class_num |
|----------------|-----------|
| Age | 2 |
| Disability_status | 2 |
| Gender_identity | 3 |
| Nationality | 6 |
| Physical_appearance | 2 |
| Race_ethnicity | 9 |
| Religion | 11 |
| SES | 2 |
| Sexual_orientation | 5 |

## Output Directory


```
├── logs/           # Training logs + per-layer evaluation curve plots
├── trace/          # Activation files + Steering Vectors
├── probe/          # Probe model weights
├── adapter/        # Adapter model weights
└── debias/         # Debias evaluation results (JSON)
```

## Anonymous Review

This repository is prepared for anonymous review. Author names, email addresses, institutional affiliations, and identifying information have been removed from the README, configuration files, comments, and paths. Model paths are represented as placeholders and should be replaced by users locally.