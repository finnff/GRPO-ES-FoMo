"""KL(policy ‖ base) over the policy's own held-out completions.

Drift is half the comparison: a method that buys its reward with a huge move
away from the base model is a different result than one that doesn't. The
estimate teacher-forces the exact sequences eval just generated through the
model twice — adapter on, adapter off — so it measures the policy where the
policy actually puts its mass.
"""

from __future__ import annotations

import torch

_SUB_ROWS = 8  # bounds the (rows, len, vocab) logits tensor


@torch.no_grad()
def _completion_logprobs(
    model, ids: torch.Tensor, attn: torch.Tensor, prompt_len: int, sub_rows: int = _SUB_ROWS
) -> torch.Tensor:
    """Per-token logprobs of the completion span [prompt_len:], (rows, gen_len).

    Logits at positions [prompt_len-1 : -1] predict the tokens at
    [prompt_len:] — the usual off-by-one of teacher forcing.
    """
    chunks = []
    for i in range(0, ids.shape[0], sub_rows):
        rows, mask = ids[i : i + sub_rows], attn[i : i + sub_rows]
        logits = model(input_ids=rows, attention_mask=mask).logits
        logprobs = logits[:, prompt_len - 1 : -1].log_softmax(dim=-1)
        targets = rows[:, prompt_len:]
        chunks.append(logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
    return torch.cat(chunks)


@torch.no_grad()
def sequence_kl_to_base(model, tok, sequences) -> float:
    """Token-weighted mean k3 KL estimate over generated sequences.

    ``sequences`` is the ``(batch_ids, prompt_len)`` list from
    ``generate(want_sequences=True)``; ``model`` is the adapter-wrapped
    PeftModel. k3 = exp(d) - d - 1 with d = logp_base - logp_policy is
    GRPO's own low-variance, non-negative estimator, so the number here is
    directly comparable to the trainer's KL term.
    """
    total_kl = 0.0
    total_tokens = 0.0
    for ids, prompt_len in sequences:
        ids = ids.to(model.device)
        # Pad id doubles as EOS, so this also drops the terminator from the
        # mask — one token per row, noise next to hundreds of content tokens.
        attn = (ids != tok.pad_token_id).long()

        lp_theta = _completion_logprobs(model, ids, attn, prompt_len)
        with model.disable_adapter():
            lp_base = _completion_logprobs(model, ids, attn, prompt_len)

        d = (lp_base - lp_theta).float()
        k3 = d.exp() - d - 1.0
        cmask = attn[:, prompt_len:].float()
        per_row = (k3 * cmask).sum(1) / cmask.sum(1).clamp_min(1.0)
        row_tokens = cmask.sum(1)

        total_kl += float((per_row * row_tokens).sum())
        total_tokens += float(row_tokens.sum())

    return total_kl / max(total_tokens, 1.0)
