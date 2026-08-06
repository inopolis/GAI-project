"""
Precise live diagnostic: generate ONE fresh greedy sample directly (bypassing
sampling_eval.py's file-write/read round trip entirely) and inspect it with
repr() to see literal newlines/whitespace, then test persistent_loop_onset
on the TRUE in-memory text.

This isolates whether the earlier offline check's newline->space join
(' '.join(body)) when reading the saved file back was ARTIFICIALLY creating
periodicity that isn't actually present in the real generated text.
"""
import sys, os
sys.path.insert(0, '.')

import torch
from src.utils import set_seed, load_json
from src.model import CharTransformerLM
from src.decoding import generate
from sampling_eval import persistent_loop_onset, TEST_PROMPTS

device = (torch.device("mps") if torch.backends.mps.is_available() else
          torch.device("cuda") if torch.cuda.is_available() else
          torch.device("cpu"))

ckpt = torch.load("runs/cosine/best.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
model = CharTransformerLM(
    vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
    n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
    n_head=cfg["n_head"], dropout=0.0).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

vocab = load_json("data_out/vocab.json")
stoi, itos = vocab["stoi"], vocab["itos"]

def encode(p):
    unk = stoi.get(" ", 0)
    return torch.tensor([[stoi.get(c, unk) for c in p]], dtype=torch.long)

def decode(ids):
    return "".join(itos.get(str(int(i)), "?") for i in ids)

prompt_name, prompt_text = TEST_PROMPTS[0]
print(f"Prompt: {prompt_text!r}\n")

set_seed(1)
idx = encode(prompt_text).to(device)
out, _ = generate(model, idx, max_new_tokens=500, temperature=0.0, top_p=1.0)
gen_ids = out[0].tolist()[len(prompt_text):]
text = decode(gen_ids)

print(f"repr() of first 300 chars (shows hidden whitespace/newlines):")
print(repr(text[:300]))
print()
print(f"Does text contain literal newline chars? {'\\n' in text}")
print(f"Newline count in first 500 chars: {text.count(chr(10))}")
print()

onset = persistent_loop_onset(text)
print(f"persistent_loop_onset(text) on the TRUE live text: {onset}")
for R in (3,4,5):
    o = persistent_loop_onset(text, P_max=60, R=R)
    print(f"  explicit R={R}: onset={o}")

# Also test with newlines stripped/replaced, matching the earlier offline
# reconstruction, to directly compare against the live (unmodified) text.
text_flattened = " ".join(text.split("\n"))
onset_flat = persistent_loop_onset(text_flattened)
print(f"\npersistent_loop_onset on NEWLINE-FLATTENED text (' '.join split by \\n): {onset_flat}")
print(f"(this mimics the earlier offline file round-trip reconstruction)")