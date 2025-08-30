import argparse, json, math, os, sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification
import importlib, subprocess, sys

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

# Prefer local copies if present
try:
    from local_runner.train_glue_imdb import load_glue, swap_attn, make_beta_trainable, compute_metrics
except Exception:
    from bert_pbfa_glue_all_azure import load_glue, swap_attn, make_beta_trainable, compute_metrics

TASKS = ["sst2", "mnli", "qqp", "qnli", "cola", "stsb", "mrpc", "rte", "wnli", "imdb"]

NUM_LABELS = {
    "sst2": 2, "mnli": 3, "qqp": 2, "qnli": 2, "cola": 2,
    "stsb": 1, "mrpc": 2, "rte": 2, "wnli": 2, "imdb": 2,
}

@torch.no_grad()
def _enable_flashattention() -> None:
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        except Exception:
            pass

def _ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        importlib.import_module(import_name)
    except Exception:
        pkg = pip_name or import_name
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', pkg])

@torch.no_grad()
def _swap_bert_attn_with(model: BertForSequenceClassification, mode: str) -> None:
    try:
        from local_runner.pbfa_attention_fast import CollapsedPBFAOptimized
    except Exception:
        from pbfa_attention_fast import CollapsedPBFAOptimized
    import torch.nn as nn
    class _PBFAWrapper(nn.Module):
        def __init__(self, orig_attn: nn.Module) -> None:
            super().__init__()
            d_model = orig_attn.self.query.weight.shape[1]
            n_heads = orig_attn.num_attention_heads
            self.attn = CollapsedPBFAOptimized(
                d_model=d_model,
                n_heads=n_heads,
                order=6,
                bias=False,
                fused_qkv=False,
                den_normalization='l2',
            )
            self.attn.q_proj.weight.data.copy_(orig_attn.self.query.weight)
            self.attn.k_proj.weight.data.copy_(orig_attn.self.key.weight)
            self.attn.v_proj.weight.data.copy_(orig_attn.self.value.weight)
            self.attn.out_proj.weight.data.copy_(orig_attn.output.dense.weight)
        def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
            out = self.attn(hidden_states)
            return (out,) if not output_attentions else (out, None)

    class _NystromWrapper(nn.Module):
        def __init__(self, orig_attn: nn.Module) -> None:
            super().__init__()
            _ensure_package('nystrom_attention', 'nystrom-attention')
            from nystrom_attention import NystromAttention
            d_model = orig_attn.self.query.weight.shape[1]
            n_heads = orig_attn.num_attention_heads
            self.attn = NystromAttention(
                dim=d_model, heads=n_heads, dim_head=d_model // n_heads, num_landmarks=256
            )
        def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
            out = self.attn(hidden_states)
            return (out,) if not output_attentions else (out, None)

    class _PerformerWrapper(nn.Module):
        def __init__(self, orig_attn: nn.Module) -> None:
            super().__init__()
            _ensure_package('performer_pytorch')
            from performer_pytorch import SelfAttention as PerformerSelfAttention
            d_model = orig_attn.self.query.weight.shape[1]
            n_heads = orig_attn.num_attention_heads
            self.attn = PerformerSelfAttention(
                dim=d_model, heads=n_heads, dim_head=d_model // n_heads, nb_features=256
            )
        def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
            out = self.attn(hidden_states)
            return (out,) if not output_attentions else (out, None)

    class _LinformerWrapper(nn.Module):
        def __init__(self, orig_attn: nn.Module) -> None:
            super().__init__()
            _ensure_package('linformer')
            from linformer import LinformerSelfAttention
            d_model = orig_attn.self.query.weight.shape[1]
            n_heads = orig_attn.num_attention_heads
            self.attn = LinformerSelfAttention(dim=d_model, heads=n_heads, k=128)
        def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
            out = self.attn(hidden_states)
            return (out,) if not output_attentions else (out, None)

    for name, module in model.named_modules():
        if module.__class__.__name__ == "BertSelfAttention":
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            if mode == 'pbfa_l2':
                setattr(parent, parts[-1], _PBFAWrapper(module))
            elif mode == 'nystrom':
                setattr(parent, parts[-1], _NystromWrapper(module))
            elif mode == 'performer':
                setattr(parent, parts[-1], _PerformerWrapper(module))
            elif mode == 'linformer':
                setattr(parent, parts[-1], _LinformerWrapper(module))
            else:
                raise ValueError("Unsupported attention mode for swap")

@torch.no_grad()
def eval_task(task: str, ckpt_path: Path, seq_len: int, dev: torch.device, batch_size: int, *, attention: str = 'pbfa_l2', iters: int = 1):
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    _, ds_val1, ds_val2 = load_glue(task, tok, seq_len)
    val_loader1 = DataLoader(ds_val1, batch_size=batch_size)
    val_loader2 = DataLoader(ds_val2, batch_size=batch_size) if ds_val2 else None

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=NUM_LABELS[task]
    )
    if attention == 'softmax_flash':
        _enable_flashattention()
    elif attention in {'pbfa_l2','nystrom','performer','linformer'}:
        _swap_bert_attn_with(model, attention)
    else:
        raise ValueError("attention must be one of {'softmax_flash','pbfa_l2'}")

    if ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.to(dev)
    model.eval()

    from torch.cuda.amp import autocast
    def _run(loader):
        all_preds, all_labels, tot_loss = [], [], 0.0
        for batch in tqdm(loader, desc=f"{task.upper()} batches", leave=False):
            if dev.type == 'cuda':
                with autocast(device_type='cuda', dtype=torch.float16):
                    out = model(
                        input_ids=batch["input_ids"].to(dev),
                        attention_mask=batch["attention_mask"].to(dev),
                        labels=batch["labels"].to(dev),
                    )
            else:
                out = model(
                    input_ids=batch["input_ids"].to(dev),
                    attention_mask=batch["attention_mask"].to(dev),
                    labels=batch["labels"].to(dev),
                )
            if task == "stsb":
                preds = out.logits.squeeze().cpu().float()
            else:
                preds = out.logits.argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(batch["labels"].cpu())
            tot_loss += out.loss.item() * batch["labels"].size(0)
        preds_cat = torch.cat(all_preds)
        labels_cat = torch.cat(all_labels)
        return compute_metrics(task, preds_cat.numpy(), labels_cat.numpy())

    last_metrics = None
    for _ in range(max(1, iters)):
        metrics1 = _run(val_loader1)
        if task == "mnli" and val_loader2 is not None:
            metrics2 = _run(val_loader2)
            last_metrics = {"acc_matched": metrics1.get("acc", 0.0), "acc_mismatched": metrics2.get("acc", 0.0)}
        else:
            last_metrics = metrics1
    return last_metrics if last_metrics is not None else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--save_csv", type=str, default="pbfa_glue_results.json")
    ap.add_argument("--task", choices=TASKS, default="mnli")
    ap.add_argument("--attention", choices=["softmax_flash","pbfa_l2","nystrom","performer","linformer"], default="pbfa_l2")
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device)
    batch_size = args.batch
    print(f"Using device: {device}.  Eval batch size = {batch_size}")

    ckpt_root = Path(args.ckpt_dir)
    ckpt_root.mkdir(exist_ok=True)

    task = args.task
    ckpt = ckpt_root / f"bert_pbfa_power_{task}.pt"
    print(f"\n=== {task.upper()} ===")
    print(f"Checkpoint: {ckpt}")
    print(f"Seq len: {args.seq}  Batch: {batch_size}  Attention: {args.attention}  iters={args.iters}")
    metrics = eval_task(task, ckpt, args.seq, device, batch_size, attention=args.attention, iters=args.iters)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
