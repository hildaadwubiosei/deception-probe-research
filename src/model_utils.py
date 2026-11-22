"""
Model loading, seeding, and the tokenize -> mask -> pad pipeline used for
fine-tuning. Loss is masked on prompt tokens so training signal focuses
only on producing the response and then stopping (EOS), not on
re-learning ordinary language modelling over the prompt.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset

MODEL_NAME = "EleutherAI/pythia-70m-deduped"


def set_all_seeds(seed=42):
    """Call this EXACTLY ONCE, at the very top of the notebook, before any
    other randomness is consumed. Do not call it again later in the same
    run - re-seeding partway through changes what every subsequent call to
    random/np.random/torch produces, which silently desynchronizes a
    notebook that gets partially re-run (this caused a real bug earlier
    in this project's history - see README "Reproducibility notes")."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fresh_model_and_tokenizer(device):
    """Always load a completely fresh pretrained checkpoint - never reuse an
    in-memory `model` variable across an experiment you intend to be a
    clean run. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(device)
    return model, tokenizer


def tokenize_and_mask(example, tokenizer):
    """Tokenize prompt+response, masking the loss (-100) on prompt tokens
    and on all tokens up to and including the response, appending EOS so
    the model learns to stop generating after the answer."""
    prompt_ids = tokenizer.encode(example["prompt"])
    response_ids = tokenizer.encode(" " + example["response"])
    eos_id = tokenizer.eos_token_id

    input_ids = prompt_ids + response_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + response_ids + [eos_id]
    return input_ids, labels


def make_batch(examples, tokenizer):
    """Pad a list of examples to the same length, with -100 labels and
    zeroed attention mask on padding positions."""
    tokenized = [tokenize_and_mask(ex, tokenizer) for ex in examples]
    max_len = max(len(ids) for ids, _ in tokenized)
    pad_id = tokenizer.pad_token_id

    batch_input_ids, batch_labels, batch_attention_mask = [], [], []
    for input_ids, labels in tokenized:
        pad_len = max_len - len(input_ids)
        batch_input_ids.append(input_ids + [pad_id] * pad_len)
        batch_labels.append(labels + [-100] * pad_len)
        batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids),
        "labels": torch.tensor(batch_labels),
        "attention_mask": torch.tensor(batch_attention_mask),
    }


class ExampleDataset(Dataset):
    """Thin wrapper so a plain list[dict] works with torch's DataLoader."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def predict_greedy(prompt, model, tokenizer, device):
    """Greedy next-token prediction right after the prompt (no generation
    loop needed since every response here is a single token)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    next_token_logits = outputs.logits[0, -1, :]
    predicted_id = torch.argmax(next_token_logits).item()
    return tokenizer.decode([predicted_id]).strip()


def overall_accuracy(examples, model, tokenizer, device):
    correct = sum(
        int(predict_greedy(ex["prompt"], model, tokenizer, device) == ex["response"].strip())
        for ex in examples
    )
    return correct / len(examples) if examples else float("nan")
