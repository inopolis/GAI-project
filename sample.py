import os
import argparse
import torch

from src.utils import set_seed, load_json
from src.model import CharTransformerLM
from src.decoding import generate


def encode_prompt(prompt: str, stoi: dict):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(ch, unk) for ch in prompt]], dtype=torch.long)


def decode_ids(ids, itos: dict):
    # ids: (T,)
    return "".join([itos[str(int(i))] if isinstance(itos, dict) and str(int(i)) in itos else itos[int(i)] for i in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_out")
    ap.add_argument("--ckpt", type=str, default="runs/baseline/best.pt")
    ap.add_argument("--prompt", type=str, default="CHAPTER 1\n")
    ap.add_argument("--max_new_chars", type=int, default=800)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)  # 0.0 => greedy
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)

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

    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    vocab_size = config["vocab_size"]

    model = CharTransformerLM(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        dropout=0.0,  # disable dropout for sampling
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    idx = encode_prompt(args.prompt, stoi)
    out = generate(
        model,
        idx=idx,
        max_new_tokens=args.max_new_chars,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )[0].tolist()

    text = decode_ids(out, itos)
    print(text)


if __name__ == "__main__":
    main()