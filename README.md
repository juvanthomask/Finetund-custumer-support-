# LoRA Customer Support Fine-Tuning

This project shows how to fine-tune a small chat model on the Bitext customer support dataset using LoRA / QLoRA. The goal is to teach the model to answer like a helpful support agent while keeping the training footprint small enough for a free Kaggle T4 GPU.

The pipeline does three main things:
1. Converts the raw Kaggle CSV into instruction-format JSONL files.
2. Trains a TinyLlama adapter with QLoRA.
3. Evaluates the fine-tuned adapter against the base model on IID and OOD examples.

## Setup

```bash
git clone https://github.com/juvanthomask/Finetund-custumer-support-.git
cd Finetund-custumer-support-
pip install -r requirements.txt
```

Download the Bitext dataset from Kaggle and place the CSV in `data/raw/`.

In Colab, the repo is assumed to live at `/content/Finetund-custumer-support-`. The scripts resolve paths from the repo root, so you can run them from anywhere once the notebook has cloned or mounted the project there.

For Kaggle Notebook use, the dataset can also be copied from `/kaggle/input/` into `data/raw/` before running the scripts.

## Project Layout

- `scripts/prepare_data.py`: builds train, validation, IID eval, and OOD eval JSONL files.
- `scripts/train.py`: trains the QLoRA adapter.
- `scripts/evaluate.py`: compares the adapter against the base model.
- `configs/qlora_config.yaml`: central training configuration.
- `notebooks/finetune_kaggle.ipynb`: Kaggle-ready end-to-end notebook.

## How to Run

### 1. Prepare data

```bash
python scripts/prepare_data.py --raw-dir data/raw --output-dir data/processed
```

This expects a CSV in `data/raw/` with the Bitext columns:
`flags, instruction, category, intent, response`.

### 2. Train

```bash
python scripts/train.py --config configs/qlora_config.yaml --data-dir data/processed --output-dir outputs
```

The script saves the adapter to `outputs/final_adapter/` and training logs to `outputs/training_logs.json`.

### 3. Evaluate

```bash
python scripts/evaluate.py --config configs/qlora_config.yaml --data-dir data/processed/eval --adapter-dir outputs/final_adapter --output-file outputs/eval_results.json
```

The evaluator reports ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, and average response length for both the base model and the fine-tuned model.

## Expected Results

You should expect the fine-tuned adapter to produce more dataset-specific support responses than the base model. On IID examples, ROUGE and BERTScore should usually improve. On OOD examples, the model should remain more stable and follow the support-agent style, even when the input is vague, angry, or multi-intent.

Exact numbers depend on the dataset split, number of training steps, and whether you run on CPU or GPU.

## LoRA and QLoRA, in Simple Terms

LoRA stands for Low-Rank Adaptation. Instead of updating every weight in the model, LoRA adds a small number of trainable adapter weights on top of the frozen base model. That makes training faster, cheaper, and easier to store, because the final artifact is just the adapter rather than a full copy of the model.

QLoRA keeps the same idea but loads the base model in 4-bit quantized form. That cuts memory use enough to fine-tune larger models on smaller GPUs. In this project, that is what makes TinyLlama practical on a free T4 card. The tradeoff is that the training setup is more sensitive to precision and batching choices, so the config is tuned conservatively for stability.

## Notes

- `outputs/` is gitignored because it can contain checkpoints and evaluation artifacts.
- `data/raw/` is gitignored because the Kaggle CSV may be large and dataset-specific.
- The scripts accept CLI arguments so you can test on a smaller subset before running a full job.
