"""Optional real pretrained model attention trace; never changes or quantizes its cache."""
import argparse
import numpy as np
from .report import export_trace


def collect(model, ids, steps=24, selected_layers=None, eos_token_id=None):
    import torch
    model.eval()
    trace, consumed = [], ids[0].tolist()
    prompt_len = ids.shape[1]
    layers = selected_layers or [len(model.model.layers) - 1]
    n_layers = len(model.model.layers)
    layers = [x if x >= 0 else n_layers + x for x in layers]
    if any(x < 0 or x >= n_layers for x in layers):
        raise ValueError("Layer index outside model")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        prefill = model(input_ids=ids, attention_mask=mask, use_cache=True)
        past = prefill.past_key_values
        next_id = prefill.logits[:, -1].argmax(-1).reshape(1, 1)
        for _ in range(steps):
            if eos_token_id is not None and int(next_id.item()) == eos_token_id:
                break
            consumed.append(int(next_id.item()))
            mask = torch.cat((mask, torch.ones_like(next_id)), dim=-1)
            output = model(input_ids=next_id, attention_mask=mask,
                           past_key_values=past, use_cache=True, output_attentions=True)
            if output.attentions is None or any(output.attentions[i] is None for i in layers):
                raise RuntimeError("Attentions missing. Load model with attn_implementation='eager'.")
            per_layer = [output.attentions[i][0, :, -1, :].float().mean(0) for i in layers]
            row = torch.stack(per_layer).mean(0).cpu().numpy()
            if len(row) != len(consumed):
                raise RuntimeError("Cache position/attention length mismatch")
            trace.append(row)
            past = output.past_key_values
            next_id = output.logits[:, -1].argmax(-1).reshape(1, 1)
    if not trace:
        raise RuntimeError("No decode queries captured; try a longer prompt or more steps")
    padded = np.full((len(trace), len(consumed)), np.nan)
    for s, row in enumerate(trace):
        padded[s, :len(row)] = row
    return padded, prompt_len, consumed, layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--prompt", default="Explain why KV cache reduces the computational cost of autoregressive decoding.")
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--layers", default="-1", help="Comma-separated layer indices; default last layer")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", default="runs/qwen")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = tokenizer.apply_chat_template([dict(role="user", content=args.prompt)],
              tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    # Keep eager attention probes short: prefill attention is quadratic in prompt length.
    if ids.shape[1] > 512:
        raise ValueError("Probe limited to 512 prompt tokens; use a shorter prompt")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
              attn_implementation="eager").to(args.device)
    a, n, tokens, layers = collect(model, ids, args.steps,
              [int(x) for x in args.layers.split(",")], tokenizer.eos_token_id)
    labels = tokenizer.convert_ids_to_tokens(tokens)
    export_trace(a, n, args.out, dict(source="PRETRAINED_MODEL", model=args.model,
       selected_layers=layers, torch=torch.__version__, transformers=transformers.__version__,
       device=args.device, dtype="float32", decoding="greedy, enable_thinking=False",
       generated_text=tokenizer.decode(tokens[n:], skip_special_tokens=True),
       limitations="Attention proxy only; cache remains full precision. No LLM quantization benchmark."), labels)
    print(f"Captured {len(a)} decode query steps -> {args.out}")


if __name__ == "__main__":
    main()
