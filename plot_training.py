import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Цвета ──────────────────────────────────────────────────────────────────────
C_TRAIN = "#2E75B6"
C_VAL   = "#E05C2A"
C_GRID  = "#EEEEEE"
C_BG    = "#FAFAFA"





def load_log(path):
    import csv
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def smooth(values, window=5):
    """Simple moving average."""
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(np.mean(values[lo:hi]))
    return out


def make_plots(log_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    rows = load_log(log_path)
    steps       = [r["step"]       for r in rows]
    train_loss  = [r["train_loss"] for r in rows]
    val_loss    = [r["val_loss"]   for r in rows]
    train_bpc   = [r["train_bpc"] for r in rows]
    val_bpc     = [r["val_bpc"]   for r in rows]

    # ── Figure 1: Loss & BPC side by side ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    fig.suptitle("Character-Level Transformer — Training Curves", fontsize=15, fontweight="bold", y=1.01)

    for ax, (train_y, val_y, ylabel, title) in zip(
        axes,
        [
            (train_loss, val_loss, "Cross-Entropy Loss (nats)", "Loss"),
            (train_bpc,  val_bpc,  "Bits Per Character (BPC)",  "BPC"),
        ]
    ):
        ax.set_facecolor(C_BG)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)

        # raw (faint)
        ax.plot(steps, train_y, color=C_TRAIN, alpha=0.18, linewidth=1, zorder=1)
        ax.plot(steps, val_y,   color=C_VAL,   alpha=0.18, linewidth=1, zorder=1)

        # smoothed
        ax.plot(steps, smooth(train_y), color=C_TRAIN, linewidth=2.2,
                label="Train", zorder=2)
        ax.plot(steps, smooth(val_y),   color=C_VAL,   linewidth=2.2,
                label="Validation", zorder=2)

        # best val marker
        best_i = int(np.argmin(val_y))
        ax.scatter([steps[best_i]], [val_y[best_i]], color=C_VAL,
                   s=80, zorder=5, label=f"Best val ({val_y[best_i]:.3f})")

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out1 = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out1}")

    # ── Figure 2: Val BPC только, крупно, с аннотациями ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)

    # random baseline: log2(vocab_size) ≈ 6.57 for vocab=95
    # We'll draw it if first val_bpc is close to that range
    random_bpc = np.log2(95)
    ax.axhline(random_bpc, color="#AAAAAA", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(steps[-1] * 0.98, random_bpc + 0.05,
            f"Random baseline ({random_bpc:.2f} BPC)", ha="right", fontsize=9, color="#888888")

    ax.plot(steps, val_bpc, color=C_GRID, linewidth=1, zorder=1)
    ax.plot(steps, smooth(val_bpc, window=7), color=C_VAL, linewidth=2.5,
            label="Validation BPC", zorder=2)

    best_i = int(np.argmin(val_bpc))
    ax.scatter([steps[best_i]], [val_bpc[best_i]], color=C_VAL,
               s=100, zorder=5)
    ax.annotate(
        f"Best: {val_bpc[best_i]:.3f} BPC\n(step {int(steps[best_i]):,})",
        xy=(steps[best_i], val_bpc[best_i]),
        xytext=(steps[best_i] + max(steps) * 0.04, val_bpc[best_i] + 0.15),
        fontsize=10, color=C_VAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_VAL, lw=1.5)
    )

    # Improvement annotation
    improvement = val_bpc[0] - val_bpc[best_i]
    ax.annotate("", xy=(steps[best_i], val_bpc[best_i]),
                xytext=(steps[0], val_bpc[0]),
                arrowprops=dict(arrowstyle="<->", color=C_TRAIN, lw=1.5))
    ax.text(steps[len(steps)//6], (val_bpc[0] + val_bpc[best_i]) / 2,
            f"−{improvement:.2f} BPC", color=C_TRAIN, fontsize=10, fontweight="bold")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Bits Per Character (BPC)", fontsize=12)
    ax.set_title("Validation BPC over Training — Baseline Model", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out2 = os.path.join(out_dir, "val_bpc_curve.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out2}")

    # ── Figure 3: Train vs Val gap (overfitting analysis) ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.set_facecolor(C_BG)
    ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)

    gap = [v - t for t, v in zip(train_bpc, val_bpc)]
    ax.fill_between(steps, 0, smooth(gap, window=7),
                    color=C_VAL, alpha=0.25, zorder=1, label="Val − Train BPC (generalisation gap)")
    ax.plot(steps, smooth(gap, window=7), color=C_VAL, linewidth=2, zorder=2)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("BPC Gap (Val − Train)", fontsize=12)
    ax.set_title("Generalisation Gap over Training", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out3 = os.path.join(out_dir, "generalization_gap.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out3}")

    print("\nВсе графики сохранены!")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="runs/baseline/log.csv",
                    help="Путь к log.csv от train.py")
    ap.add_argument("--out", type=str, default="runs/baseline/plots",
                    help="Папка куда сохранить PNG")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"Файл не найден: {args.log}")
        print("Укажи путь явно: python3 plot_training.py --log путь/к/log.csv")
        exit(1)

    make_plots(args.log, args.out)
    