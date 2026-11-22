"""
Activation extraction.

Design decision (see README): activations are extracted at PROMPT-TIME, at
the LAST token of the prompt, before any generation - not during or after
generation. This captures the model's state at the moment it has all
information needed to decide how to respond, before that decision is
"spent" on producing a specific output token. All 7 hidden-state layers
are kept (0 = embedding output, 1-6 = the six transformer layers).
"""

import numpy as np
import torch


def extract_activations(prompt, model, tokenizer, device):
    """Returns {layer_idx: np.array[512]} - the last prompt token's hidden
    state at every layer, for one prompt."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    activations = {}
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        activations[layer_idx] = hidden_state[0, -1, :].cpu().numpy()
    return activations


def extract_dataset_activations(examples, model, tokenizer, device, n_layers=7):
    """Returns {layer_idx: np.array[n_examples, 512]} for a whole dataset."""
    n = len(examples)
    layer_activations = {layer: np.zeros((n, 512), dtype=np.float32) for layer in range(n_layers)}
    for i, ex in enumerate(examples):
        acts = extract_activations(ex["prompt"], model, tokenizer, device)
        for layer in range(n_layers):
            layer_activations[layer][i] = acts[layer]
    return layer_activations
