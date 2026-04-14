"""
train_cosine.py — train.py with cosine LR schedule + weight tying


baseline train.py:
  1. Cosine LR schedule с linear warmup (--warmup_steps)
  2. flag --tie_weights: tie input/output embeddings (меньше параметров)

running:
  python3 train_cosine.py \
    --data_dir data_out \
    --out_dir runs/cosine \
    --max_steps 20000 \
    --warmup_steps 500
"""

import os
import time
import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, load_json, save_json, bpc_from_loss
from src.dataset import CharBinDataset
from src.model import CharTransformerLM


def get_lr(step, max_steps, warmup_steps, lr_max, lr_min=1e-5):
    """Cosine decay с linear warmup."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step > max_steps:
        return lr_min
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + cosine * (lr_max - lr_min)


@torch.no_grad()
def estimate_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",     type=str,   default="data_out")
    ap.add_argument("--out_dir",      type=str,   default="runs/cosine")
    ap.add_argument("--seed",         type=int,   default=1337)

    # model
    ap.add_argument("--block_size",   type=int,   default=256)
    ap.add_argument("--n_layer",      type=int,   default=4)
    ap.add_argument("--n_embd",       type=int,   default=128)
    ap.add_argument("--n_head",       type=int,   default=4)
    ap.add_argument("--dropout",      type=float, default=0.1)
    ap.add_argument("--tie_weights",  action="store_true",
                    help="Tie input/output embeddings (reduces params, improves generalisation)")

    # training
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=3e-4)
    ap.add_argument("--lr_min",       type=float, default=1e-5,
                    help="Minimum LR at end of cosine decay")
    ap.add_argument("--warmup_steps", type=int,   default=500,
                    help="Linear warmup steps before cosine decay begins")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_steps",    type=int,   default=20000)
    ap.add_argument("--eval_every",   type=int,   default=500)
    ap.add_argument("--eval_batches", type=int,   default=50)
    ap.add_argument("--grad_clip",    type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    vocab      = load_json(os.path.join(args.data_dir, "vocab.json"))
    vocab_size = len(vocab["stoi"])
    print("Vocab size:", vocab_size)
    dtype = np.uint16 if vocab_size < 65535 else np.uint32

    train_path = os.path.join(args.data_dir, "train.bin")
    val_path   = os.path.join(args.data_dir, "val.bin")
    test_path  = os.path.join(args.data_dir, "test.bin")

    train_ds = CharBinDataset(train_path, block_size=args.block_size, dtype=dtype)
    val_ds   = CharBinDataset(val_path,   block_size=args.block_size, dtype=dtype)
    test_ds  = CharBinDataset(test_path,  block_size=args.block_size, dtype=dtype)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = CharTransformerLM(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    # ── Weight tying: share token embedding и output head ─────────────────
    if args.tie_weights:
        model.head.weight = model.tok_emb.weight
        print("Weight tying enabled (tok_emb == head).")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args)
    config["vocab_size"] = vocab_size
    config["total_params"] = total_params
    save_json(config, os.path.join(args.out_dir, "config.json"))

    best_val  = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")
    log_path  = os.path.join(args.out_dir, "log.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("step,lr,train_loss,val_loss,train_bpc,val_bpc,seconds\n")

    t0   = time.time()
    step = 0
    pbar = tqdm(total=args.max_steps, desc="training")
    it   = iter(train_loader)

    model.train()
    while step < args.max_steps:
        # ── LR schedule ───────────────────────────────────────────────────
        lr_now = get_lr(step, args.max_steps, args.warmup_steps, args.lr, args.lr_min)
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

        step += 1
        pbar.update(1)

        if step % args.eval_every == 0 or step == 1:
            train_loss = float(loss.item())
            val_loss   = estimate_loss(model, val_loader, device, args.eval_batches)
            train_bpc  = bpc_from_loss(train_loss)
            val_bpc    = bpc_from_loss(val_loss)
            secs       = time.time() - t0

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{step},{lr_now:.6f},{train_loss:.6f},{val_loss:.6f},{train_bpc:.4f},{val_bpc:.4f},{secs:.1f}\n")

            pbar.set_postfix(
                lr=f"{lr_now:.1e}",
                train_loss=f"{train_loss:.3f}",
                val_loss=f"{val_loss:.3f}",
                val_bpc=f"{val_bpc:.3f}",
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "config": config}, best_path)

    pbar.close()

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss = estimate_loss(model, test_loader, device, args.eval_batches)
    test_bpc  = bpc_from_loss(test_loss)

    result = {
        "best_val_loss": best_val,
        "best_val_bpc":  bpc_from_loss(best_val),
        "test_loss":     test_loss,
        "test_bpc":      test_bpc,
    }
    save_json(result, os.path.join(args.out_dir, "final_metrics.json"))
    print("\nFinal metrics:", result)
    print(f"Checkpoint: {best_path}")
    print(f"Log:        {log_path}")


if __name__ == "__main__":
    main()