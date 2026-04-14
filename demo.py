"""
demo.py — Quick interactive demo for the final presentation.

Shows greedy vs temperature=0.8 vs nucleus p=0.95 side-by-side
for any prompt the user types.

Usage: Run this command in the terminal, then follow the prompts
    python3 demo.py --ckpt runs/cosine/best.pt
    python3 demo.py --ckpt runs/baseline/best.pt --chars 400
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import set_seed, load_json
from src.model import CharTransformerLM
from src.decoding import generate


STRATEGIES = [
    {"label": "Greedy (temp=0.0)",        "temperature": 0.0, "top_k": 0,  "top_p": 1.0},
    {"label": "Temperature 0.8",           "temperature": 0.8, "top_k": 0,  "top_p": 1.0},
    {"label": "Nucleus p=0.95, temp=1.0",  "temperature": 1.0, "top_k": 0,  "top_p": 0.95},
]

WIDTH = 72


def banner(text, char="═"):
    pad = max(0, WIDTH - len(text) - 4)
    return f"╔══ {text} {'═' * pad}╗"

def footer():
    return "╚" + "═" * (WIDTH - 1) + "╝"

def box_text(text, width=WIDTH - 4):
    """Wrap text into box lines."""
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        while len(raw_line) > width:
            lines.append(raw_line[:width])
            raw_line = raw_line[width:]
        lines.append(raw_line)
    return lines


def encode_prompt(prompt, stoi):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(ch, unk) for ch in prompt]], dtype=torch.long)


def decode_ids(ids, itos):
    return "".join([itos[str(int(i))] if str(int(i)) in itos else "?" for i in ids])


def load_model(ckpt_path, device):
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model  = CharTransformerLM(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def run_demo(model, stoi, itos, prompt, n_chars, seed=42):
    print()
    print(f"  Prompt: \"{prompt.strip()}\"")
    print(f"  Generating {n_chars} characters per strategy...\n")

    for cfg in STRATEGIES:
        set_seed(seed)
        idx = encode_prompt(prompt, stoi).to(next(model.parameters()).device)
        out = generate(
            model, idx,
            max_new_tokens=n_chars,
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
        )[0].tolist()
        text = decode_ids(out, itos)[len(prompt):]  # strip prompt from output

        print(banner(cfg["label"]))
        for line in box_text(text):
            print(f"║  {line:<{WIDTH-4}}  ║" if line else f"║{'':^{WIDTH-2}}║")
        print(footer())
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",     type=str, default="runs/cosine/best.pt")
    ap.add_argument("--data_dir", type=str, default="data_out")
    ap.add_argument("--chars",    type=int, default=350,
                    help="Characters to generate per strategy")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--prompt",   type=str, default=None,
                    help="Fixed prompt (skips interactive mode)")
    args = ap.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt}")
    model, config = load_model(args.ckpt, device)
    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    stoi, itos = vocab["stoi"], vocab["itos"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['n_layer']}L × {config['n_embd']}d × {config['n_head']}H  "
          f"| {n_params:,} params | vocab {config['vocab_size']} | device {device}")

    # ── Prompts ───────────────────────────────────────────────────────────
    DEFAULT_PROMPTS = [
        "CHAPTER 1\n",
        "The night was ",
        "She had never ",
        "It was the best of ",
    ]

    if args.prompt:
        # Single run, non-interactive
        run_demo(model, stoi, itos, args.prompt, args.chars, args.seed)
        return

    # Interactive loop
    print("\n" + "─" * WIDTH)
    print("  CHARACTER-LEVEL TRANSFORMER — LIVE DEMO")
    print("  Comparing: Greedy | Temperature 0.8 | Nucleus p=0.95")
    print("─" * WIDTH)
    print("\nDefault prompts:")
    for i, pr in enumerate(DEFAULT_PROMPTS, 1):
        print(f"  [{i}] \"{pr.strip()}\"")
    print("  [c] Custom prompt")
    print("  [q] Quit")

    while True:
        print()
        choice = input("Choose [1-4 / c / q]: ").strip().lower()

        if choice == "q":
            print("Bye!")
            break
        elif choice == "c":
            prompt = input("Enter prompt: ")
            if not prompt:
                continue
        elif choice in {"1", "2", "3", "4"}:
            prompt = DEFAULT_PROMPTS[int(choice) - 1]
        else:
            print("Invalid choice.")
            continue

        run_demo(model, stoi, itos, prompt, args.chars, args.seed)

        again = input("Generate again? [y/n]: ").strip().lower()
        if again != "y":
            print("Bye!")
            break


if __name__ == "__main__":
    main()
