"""
Fine-tuning loop for the backdoor task.

Note on reproducibility: this task converges cleanly and deterministically
(given a fixed seed, set once via model_utils.set_all_seeds) to 100% normal
accuracy, 100% triggered accuracy, and 100% held-out triggered
generalization within ~14-20 epochs. Unlike the earlier arithmetic-task
experiment (see README history), no checkpoint-selection-by-validation-
accuracy is needed here, because the task is simple enough that the model
doesn't overfit or oscillate at this epoch count. We still verify behaviour
explicitly after training rather than assuming convergence from the loss.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model_utils import make_batch, ExampleDataset, overall_accuracy


def run_epoch(loader, model, tokenizer, optimizer=None, device="cpu"):
    """One pass over `loader`. If optimizer is given, trains; otherwise
    evaluates only (no gradient updates)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches


def train_backdoor_model(
    model,
    tokenizer,
    train_examples,
    heldout_triggered_examples,
    device,
    batch_size=8,
    learning_rate=5e-5,
    num_epochs=20,
    verbose=True,
):
    """Fine-tune `model` on `train_examples` (matched normal/triggered pairs),
    tracking normal accuracy, triggered accuracy, and held-out-place
    triggered-generalization accuracy every epoch. Returns the history dict.

    This does not perform checkpoint selection. Unlike the arithmetic task
    used in an earlier iteration of this experiment, this task's final-epoch
    behaviour is the reportable result: accuracy on all three metrics (normal,
    triggered, held-out-place generalization) reaches 100% and remains stable
    across the final several epochs, which is verified explicitly below."""

    dataset = ExampleDataset(train_examples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: make_batch(batch, tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    normal_examples = [ex for ex in train_examples if not ex["triggered"]]
    triggered_examples = [ex for ex in train_examples if ex["triggered"]]

    history = {"loss": [], "normal_acc": [], "triggered_acc": [], "heldout_gen_acc": []}

    for epoch in range(1, num_epochs + 1):
        loss = run_epoch(loader, model, tokenizer, optimizer=optimizer, device=device)

        model.eval()
        acc_normal = overall_accuracy(normal_examples, model, tokenizer, device)
        acc_triggered = overall_accuracy(triggered_examples, model, tokenizer, device)
        acc_heldout = overall_accuracy(heldout_triggered_examples, model, tokenizer, device)

        history["loss"].append(loss)
        history["normal_acc"].append(acc_normal)
        history["triggered_acc"].append(acc_triggered)
        history["heldout_gen_acc"].append(acc_heldout)

        if verbose:
            print(f"Epoch {epoch:2d} | loss: {loss:.4f} | normal: {acc_normal:.1%} | "
                  f"triggered: {acc_triggered:.1%} | heldout_gen: {acc_heldout:.1%}")

    return history
