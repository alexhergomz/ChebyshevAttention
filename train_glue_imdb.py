import argparse, math, os, time, warnings, importlib, json, random
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
import importlib, subprocess, sys

try:
    from datasets import load_dataset
except Exception:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'datasets'])
    from datasets import load_dataset


def _ensure_lion_cls():
    try:
        mod = importlib.import_module('lion_pytorch')
        return getattr(mod, 'Lion')
    except Exception:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'lion-pytorch'])
            mod = importlib.import_module('lion_pytorch')
            return getattr(mod, 'Lion')
        except Exception:
            class _Lion(torch.optim.Optimizer):
                def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                    super().__init__(params, defaults)
                @torch.no_grad()
                def step(self):
                    for group in self.param_groups:
                        lr = group['lr']; beta1, beta2 = group['betas']; wd = group.get('weight_decay', 0.0)
                        for p in group['params']:
                            if p.grad is None: continue
                            g = p.grad
                            state = self.state[p]
                            if 'exp_avg' not in state:
                                state['exp_avg'] = torch.zeros_like(p)
                            m = state['exp_avg']
                            m.mul_(beta2).add_(g, alpha=1 - beta2)
                            upd = m.sign().mul(lr)
                            if wd != 0: p.add_(p, alpha=-lr * wd)
                            p.add_(upd, alpha=-beta1).add_(g.sign().mul(lr), alpha=-(1 - beta1))
            return _Lion


def load_glue(task: str, tok, seq_len: int):
    if task == "imdb":
        ds = load_dataset("imdb")
        def tok_batch(batch):
            enc = tok(batch["text"], truncation=True, max_length=seq_len, padding="max_length", return_token_type_ids=True)
            enc["labels"] = batch["label"]
            return enc
        cols_to_remove = [c for c in ds["train"].column_names if c not in {"label"}]
        ds_tok = ds.map(tok_batch, batched=True, remove_columns=cols_to_remove)
        ds_tok.set_format(type="torch")
        return ds_tok["train"], ds_tok["test"], None
    ds = load_dataset("glue", task)
    if task == "qqp":
        col_a, col_b = "question1", "question2"
    elif task == "qnli":
        col_a, col_b = "question", "sentence"
    elif task == "mnli":
        col_a, col_b = "premise", "hypothesis"
    elif task in {"mrpc", "stsb", "rte", "wnli"}:
        col_a, col_b = "sentence1", "sentence2"
    else:
        col_a, col_b = "sentence", None
    def tok_batch(batch):
        if col_b is not None:
            enc = tok(batch[col_a], batch[col_b], truncation=True, max_length=seq_len, padding="max_length", return_token_type_ids=True)
        else:
            enc = tok(batch[col_a], truncation=True, max_length=seq_len, padding="max_length", return_token_type_ids=True)
        if task == "stsb":
            enc["labels"] = [float(x) for x in batch["label"]]
        else:
            enc["labels"] = batch["label"]
        return enc
    cols_to_remove = [c for c in ds["train"].column_names if c not in {"label"}]
    ds_tok = ds.map(tok_batch, batched=True, remove_columns=cols_to_remove)
    ds_tok.set_format(type="torch")
    if task == "mnli":
        return ds_tok["train"], ds_tok["validation_matched"], ds_tok["validation_mismatched"]
    else:
        return ds_tok["train"], ds_tok["validation"], None


def compute_metrics(task: str, preds, labels) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from scipy.stats import pearsonr
    metrics = {}
    if task == "stsb":
        metrics["pearson"] = pearsonr(preds, labels)[0]
    elif task in {"mrpc", "qqp"}:
        metrics.update(acc=accuracy_score(labels, preds), f1=f1_score(labels, preds))
    elif task == "cola":
        metrics["mcc"] = matthews_corrcoef(labels, preds)
    else:
        metrics["acc"] = accuracy_score(labels, preds)
    return metrics


def swap_attn(model, mode: str):
    # softmax_flash keeps baseline
    if mode == 'softmax_flash':
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        except Exception:
            pass
        return
    import torch.nn as nn
    try:
        from local_runner.pbfa_attention_fast import CollapsedPBFAOptimized
    except Exception:
        from pbfa_attention_fast import CollapsedPBFAOptimized
    class BertSelfAttentionPBFAFast(nn.Module):
        def __init__(self, orig):
            super().__init__()
            d_model = orig.self.query.weight.shape[1]
            n_heads = orig.num_attention_heads
            self.pbfa = CollapsedPBFAOptimized(d_model, n_heads, order=6, fused_qkv=False, den_normalization='l2')
            self.pbfa.q_proj.weight.data.copy_(orig.self.query.weight)
            self.pbfa.k_proj.weight.data.copy_(orig.self.key.weight)
            self.pbfa.v_proj.weight.data.copy_(orig.self.value.weight)
            self.pbfa.out_proj.weight.data.copy_(orig.output.dense.weight)
        def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
            ctx = self.pbfa(hidden_states)
            return (ctx,) if not output_attentions else (ctx, None)
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BertSelfAttention":
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], BertSelfAttentionPBFAFast(module))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sst2","mnli","qqp","qnli","imdb"], default="sst2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--effective_batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--optimizer", choices=["lion","adamw"], default="lion")
    ap.add_argument("--attention", choices=["softmax_flash","pbfa_l2"], default="pbfa_l2")
    args, _ = ap.parse_known_args()

    device = torch.device(args.device)
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ds_train, ds_val1, ds_val2 = load_glue(args.task, tok, args.seq)
    train_loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
    val_loader1 = DataLoader(ds_val1, batch_size=args.batch)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=(3 if args.task=="mnli" else 2)).to(device)
    swap_attn(model, args.attention)

    accum_steps = max(1, args.effective_batch // args.batch)
    if args.optimizer == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        Lion = _ensure_lion_cls()
        optim = Lion(model.parameters(), lr=args.lr)

    from transformers.optimization import get_linear_schedule_with_warmup
    total_steps = (len(train_loader) * args.epochs) // accum_steps
    sched = get_linear_schedule_with_warmup(optim, max(1, int(args.warmup_pct * total_steps)), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    model.train()
    for ep in range(args.epochs):
        for step, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch.get("token_type_ids", None).to(device) if "token_type_ids" in batch else None,
                    labels=batch["labels"].to(device),
                )
                loss = out.loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optim); scaler.update(); sched.step(); optim.zero_grad()
        print(f"epoch {ep+1}/{args.epochs} done")

if __name__ == "__main__":
    main()
