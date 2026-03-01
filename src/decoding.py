import torch


def top_k_filtering(logits: torch.Tensor, k: int):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, k)
    thresh = v[..., -1, None]
    return torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)


def top_p_filtering(logits: torch.Tensor, p: float):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)

    # mask tokens where cumulative prob exceeds p
    mask = cum > p
    # keep at least 1 token
    mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    # unsort back
    unsorted = torch.empty_like(sorted_logits).scatter(-1, sorted_idx, sorted_logits)
    return unsorted


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0):
    """
    idx: (B,T) int64
    """
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]  # crop context
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # last token

        if temperature == 0.0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = top_k_filtering(logits, top_k)
            logits = top_p_filtering(logits, top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    return idx