import os
import sys
import argparse
import math
import torch
import numpy as np
from collections import Counter

# ── добавить src/ в путь ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import set_seed, load_json
from src.model import CharTransformerLM
from src.decoding import generate


# ── Метрики ─────────────────────────────────────────────────────────────────

def type_token_ratio(text):
    """Лексическое разнообразие: уникальных слов / всего слов."""
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def char_ngram_entropy(text, n=4):
    """Энтропия символьных n-грамм — мера разнообразия на уровне символов."""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def repetition_rate(text, n=5):
    """Доля повторяющихся n-грамм (0 = нет повторений, 1 = всё повторяется)."""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def encode_prompt(prompt, stoi):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(ch, unk) for ch in prompt]], dtype=torch.long)


def decode_ids(ids, itos):
    return "".join([itos[str(int(i))] if str(int(i)) in itos else "?" for i in ids])


# ── Конфиги сэмплирования ────────────────────────────────────────────────────

CONFIGS = [
    {"name": "greedy",           "temperature": 0.0, "top_k": 0,  "top_p": 1.0},
    {"name": "temp_0.6",         "temperature": 0.6, "top_k": 0,  "top_p": 1.0},
    {"name": "temp_0.8",         "temperature": 0.8, "top_k": 0,  "top_p": 1.0},
    {"name": "temp_1.0",         "temperature": 1.0, "top_k": 0,  "top_p": 1.0},
    {"name": "top_k_10",         "temperature": 1.0, "top_k": 10, "top_p": 1.0},
    {"name": "top_k_50",         "temperature": 1.0, "top_k": 50, "top_p": 1.0},
    {"name": "nucleus_p0.9",     "temperature": 1.0, "top_k": 0,  "top_p": 0.9},
    {"name": "nucleus_p0.95",    "temperature": 1.0, "top_k": 0,  "top_p": 0.95},
    {"name": "temp0.8_p0.9",     "temperature": 0.8, "top_k": 0,  "top_p": 0.9},
]

PROMPTS = [
    ("chapter", "CHAPTER 1\n"),
    ("night",   "The night was "),
    ("she",     "She had never "),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_out")
    ap.add_argument("--ckpt",     type=str, default="runs/baseline/best.pt")
    ap.add_argument("--out_dir",  type=str, default="runs/baseline/sampling_eval")
    ap.add_argument("--n_chars",  type=int, default=600,
                    help="Сколько символов генерировать для каждого сэмпла")
    ap.add_argument("--n_seeds",  type=int, default=3,
                    help="Сколько разных seed усреднять для метрик")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Загрузка модели ────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi = vocab["stoi"]
    itos = vocab["itos"]

    ckpt   = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]

    model = CharTransformerLM(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Таблица результатов ────────────────────────────────────────────────
    rows = []  # список словарей для CSV
    samples_path = os.path.join(args.out_dir, "samples.txt")

    with open(samples_path, "w", encoding="utf-8") as sf:
        sf.write("=" * 80 + "\n")
        sf.write("SAMPLING EVALUATION — GENERATED SAMPLES\n")
        sf.write("=" * 80 + "\n\n")

        for cfg in CONFIGS:
            print(f"\n── {cfg['name']} ──")
            ttr_list, entropy_list, rep_list = [], [], []

            for prompt_name, prompt_text in PROMPTS:
                seed_ttr, seed_ent, seed_rep = [], [], []

                for seed in range(1, args.n_seeds + 1):
                    set_seed(seed)
                    idx = encode_prompt(prompt_text, stoi).to(device)
                    out = generate(
                        model, idx,
                        max_new_tokens=args.n_chars,
                        temperature=cfg["temperature"],
                        top_k=cfg["top_k"],
                        top_p=cfg["top_p"],
                    )[0].tolist()
                    text = decode_ids(out, itos)
                    generated_only = text[len(prompt_text):]

                    seed_ttr.append(type_token_ratio(generated_only))
                    seed_ent.append(char_ngram_entropy(generated_only, n=4))
                    seed_rep.append(repetition_rate(generated_only, n=5))

                # Сохранить один сэмпл (seed=1) для каждого промпта
                set_seed(1)
                idx = encode_prompt(prompt_text, stoi).to(device)
                out = generate(
                    model, idx,
                    max_new_tokens=args.n_chars,
                    temperature=cfg["temperature"],
                    top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                )[0].tolist()
                best_sample = decode_ids(out, itos)

                sf.write(f"[{cfg['name']}] prompt='{prompt_text.strip()}'\n")
                sf.write("-" * 60 + "\n")
                sf.write(best_sample + "\n\n")

                ttr_list.append(np.mean(seed_ttr))
                entropy_list.append(np.mean(seed_ent))
                rep_list.append(np.mean(seed_rep))

            row = {
                "strategy":    cfg["name"],
                "temperature": cfg["temperature"],
                "top_k":       cfg["top_k"],
                "top_p":       cfg["top_p"],
                "avg_ttr":     round(float(np.mean(ttr_list)), 4),
                "avg_4gram_entropy": round(float(np.mean(entropy_list)), 4),
                "avg_rep_rate": round(float(np.mean(rep_list)), 4),
            }
            rows.append(row)
            print(f"  TTR={row['avg_ttr']:.3f}  entropy={row['avg_4gram_entropy']:.2f}  rep={row['avg_rep_rate']:.3f}")

    # ── Сохранить CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nМетрики сохранены → {csv_path}")
    print(f"Сэмплы сохранены  → {samples_path}")

    # ── Красивая таблица в терминал ───────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Strategy':<20} {'Temp':>5} {'TopK':>5} {'TopP':>5} {'TTR':>6} {'Entropy':>8} {'RepRate':>8}")
    print("-" * 75)
    for r in rows:
        print(f"{r['strategy']:<20} {r['temperature']:>5} {r['top_k']:>5} {r['top_p']:>5} "
              f"{r['avg_ttr']:>6.3f} {r['avg_4gram_entropy']:>8.2f} {r['avg_rep_rate']:>8.3f}")
    print("=" * 75)

    # ── Построить график метрик ────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names    = [r["strategy"] for r in rows]
        ttrs     = [r["avg_ttr"] for r in rows]
        entropies = [r["avg_4gram_entropy"] for r in rows]
        reps     = [r["avg_rep_rate"] for r in rows]

        x = np.arange(len(names))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
        fig.suptitle("Sampling Strategy Comparison", fontsize=14, fontweight="bold")

        colors = ["#E05C2A" if "greedy" in n else "#2E75B6" for n in names]

        for ax, vals, title, ylabel in zip(
            axes,
            [ttrs, entropies, reps],
            ["Type-Token Ratio (↑ more diverse)", "4-gram Entropy (↑ more diverse)", "Repetition Rate (↓ better)"],
            ["TTR", "Entropy (bits)", "Rep Rate"]
        ):
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_facecolor("#FAFAFA")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, "sampling_metrics.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"График метрик → {plot_path}")
    except Exception as e:
        print(f"Не удалось построить график: {e}")




if __name__ == "__main__":
    main()