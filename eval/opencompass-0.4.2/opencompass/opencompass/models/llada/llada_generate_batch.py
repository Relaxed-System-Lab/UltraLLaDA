## copy from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model,
             prompt,
             steps=128,
             gen_length=128,
             block_length=128,
             temperature=0.,
             cfg_scale=0.,
             remasking='low_confidence',
             mask_id=126336):
    """
    Batched LLaDA sampling.

    Args:
        model: Mask predictor.
        prompt: Tensor of shape (B, L) or (L,). If 1-D, it will be unsqueezed to (1, L).
        steps: Sampling steps (per whole sequence), must be divisible by num_blocks.
        gen_length: Total number of tokens to generate.
        block_length: Block length for semi-autoregressive remasking. Must divide gen_length.
        temperature: Gumbel-Max temperature (0 means argmax on logits).
        cfg_scale: Classifier-free guidance scale (unsupervised).
        remasking: 'low_confidence' or 'random'.
        mask_id: Token id used as [MASK] (default 126336).
    Returns:
        x: Tensor of shape (B, L + gen_length) containing prompt + generated tokens.
    """
    # Normalize prompt to [B, L]
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    B, L = prompt.shape

    device = prompt.device if hasattr(prompt, 'device') else next(model.parameters()).device
    total_len = L + gen_length

    # Initialize with [MASK] then place the prompt
    x = torch.full((B, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompt
    prompt_index = (x != mask_id)  # True only at prompt positions initially

    # Sanity checks for block scheduling
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        # The current block region we are filling this phase:
        blk_start = L + num_block * block_length
        blk_end   = L + (num_block + 1) * block_length

        # How many tokens should be transferred per step for this block (per sample)
        block_mask_index = (x[:, blk_start:blk_end] == mask_id)  # [B, block_length]
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # [B, steps_per_block]

        for i in range(steps_per_block):
            mask_index = (x == mask_id)  # [B, total_len]

            # ---- CFG (batched) ----
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id  # remove prompt info for UNCOND branch
                x_cat = torch.cat([x, un_x], dim=0)  # [2B, total_len]
                logits_cat = model(x_cat).logits      # [2B, total_len, V]
                logits, un_logits = torch.chunk(logits_cat, 2, dim=0)  # each [B, total_len, V]
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits  # [B, total_len, V]

            # ---- Sample proposal x0 via (optionally) Gumbel-Max ----
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)  # [B, total_len, V]
            x0 = torch.argmax(logits_with_noise, dim=-1)                           # [B, total_len]

            # ---- Confidence (for remasking) ----
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)  # [B, total_len, V]
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=x0.unsqueeze(-1)), -1)  # [B, total_len]
            elif remasking == 'random':
                x0_p = torch.rand((B, total_len), device=device)
            else:
                raise NotImplementedError(remasking)

            # Do not consider positions beyond the current block this phase
            x0_p[:, blk_end:] = -np.inf  # only allow choosing within [blk_start:blk_end) this phase

            # Keep already-filled tokens (prompt & past decided tokens) frozen
            x0 = torch.where(mask_index, x0, x)                  # only update where masked
            confidence = torch.where(mask_index, x0_p, -np.inf)  # only consider masked positions

            # ---- Pick per-sample top-k positions to transfer at this step ----
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)  # [B, total_len]
            for j in range(B):
                k = int(num_transfer_tokens[j, i].item())
                if k <= 0:
                    continue
                # Top-k across the whole seq, but only positions with finite confidence are candidates
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True

            # ---- Commit proposed tokens ----
            x[transfer_index] = x0[transfer_index]

    return x



def main():
    device = 'cuda'

    path = 'GSAI-ML/LLaDA-8B-Instruct'

    model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
