# Local Runner (no Azure)

This folder contains standalone wrappers to fine-tune and evaluate BERT with selectable attention mechanisms on GLUE or IMDB, without any Azure-specific setup.

## Prerequisites
- Python 3.9+
- CUDA GPU recommended (for FlashAttention via PyTorch SDPA and AMP)

Install basics:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm scikit-learn scipy
```
Optional attentions (installed on-demand by the scripts if missing):
```bash
pip install lion-pytorch nystrom-attention performer-pytorch linformer
```

## Evaluate (single task/attention)
```bash
python local_runner/local_eval.py --task mnli --attention pbfa_l2 --seq 128 --batch 16 --iters 1 --device cuda
```
- attentions: softmax_flash | pbfa_l2 | nystrom | performer | linformer
- tasks: sst2, mnli, qqp, qnli, cola, stsb, mrpc, rte, wnli, imdb

## Train (fine-tune from bert-base-uncased)
```bash
python local_runner/local_train.py --task mnli --attention softmax_flash --optimizer lion --epochs 3 --seq 128 --batch 16 --device cuda
```
- optimizers: lion | adamw (Lion auto-installed if missing)
- AMP is enabled on CUDA by default

##Install

- sudo apt update && sudo apt install -y python3-venv git && git clone https://github.com/alexhergomz/ChebyshevAttention.git && cd ChebyshevAttention && python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && pip install transformers datasets accelerate evaluate
