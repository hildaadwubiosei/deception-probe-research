"""
Evaluation: behavioural labeling of control-suite examples, and the
confound-analysis statistics (per-layer AUROC against multiple targets,
and place-resampled bootstrap confidence intervals).

CRITICAL METHODOLOGY NOTE (see README "Reproducibility notes"):
Behavioural labels MUST always reflect the model's ACTUAL generated output
at the time of evaluation - never an assumed or previously-recorded label.
Earlier in this project's history, a partial/incomplete recording pass left
stale labels in a dataset dict, which silently produced wrong downstream
AUROC numbers for several turns before being caught. record_behavior()
below is meant to be the ONLY way labels are produced, called fresh, in one
uninterrupted pass, every time a dataset's behaviour needs to be known.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from .model_utils import predict_greedy
from .data_generation import TRIGGER, WRONG_ANSWER_FACT


def record_behavior(examples, model, tokenizer, device):
    """Run the model on every example and record its ACTUAL output and a
    3-way behavioural classification: backdoor / normal / other. Mutates
    and returns `examples`. Always call this in one single, uninterrupted
    pass over the WHOLE list - do not call it partially or resume a
    previous partial pass, and do not assume a label without calling this."""
    for ex in examples:
        actual = predict_greedy(ex["prompt"], model, tokenizer, device)
        ex["actual_response"] = actual
        if actual == WRONG_ANSWER_FACT:
            ex["observed_behavior"] = "backdoor"
        elif "expected_normal_answer" in ex and actual == ex["expected_normal_answer"]:
            ex["observed_behavior"] = "normal"
        elif "response" in ex and actual == ex["response"]:
            ex["observed_behavior"] = "normal"
        else:
            ex["observed_behavior"] = "other"
    return examples


def behavior_breakdown(examples, control_types=("position", "inert_mention", "novel_marker")):
    """Returns a dict of {control_type: {backdoor, normal, other, n}} counts."""
    breakdown = {}
    for ctype in control_types:
        subset = [ex for ex in examples if ex.get("control_type") == ctype]
        n = len(subset)
        breakdown[ctype] = {
            "n": n,
            "backdoor": sum(1 for ex in subset if ex["observed_behavior"] == "backdoor"),
            "normal": sum(1 for ex in subset if ex["observed_behavior"] == "normal"),
            "other": sum(1 for ex in subset if ex["observed_behavior"] == "other"),
        }
    return breakdown


def targets_from_examples(examples):
    """Build the three analysis targets used throughout: trigger_present,
    backdoor_target (observed_behavior == 'backdoor'), normal_target
    (observed_behavior == 'normal')."""
    trigger_present = np.array([1 if TRIGGER in ex["prompt"] else 0 for ex in examples])
    backdoor_target = np.array([1 if ex["observed_behavior"] == "backdoor" else 0 for ex in examples])
    normal_target = np.array([1 if ex["observed_behavior"] == "normal" else 0 for ex in examples])
    return trigger_present, backdoor_target, normal_target


def per_layer_auroc_vs_targets(layer_probs, trigger_present, backdoor_target, normal_target):
    """layer_probs: {layer: np.array of predicted probabilities}.
    Returns a dict {layer: {"vs_trigger": auroc, "vs_backdoor": auroc, "vs_normal": auroc_or_nan}}."""
    results = {}
    for layer, probs in layer_probs.items():
        row = {"vs_trigger": roc_auc_score(trigger_present, probs),
               "vs_backdoor": roc_auc_score(backdoor_target, probs)}
        row["vs_normal"] = roc_auc_score(normal_target, probs) if len(set(normal_target)) > 1 else float("nan")
        results[layer] = row
    return results


def bootstrap_auroc_by_place(examples, probs, target, n_boot=1000, seed=0):
    """Bootstrap AUROC CI, resampling by the 'place' field (not by
    individual prompt) - multiple templates for the same place are not
    independent observations. Returns the array of bootstrap AUROCs
    (use np.percentile([2.5, 97.5]) for a 95% CI)."""
    rng = np.random.RandomState(seed)
    unique_places = list(set(ex["place"] for ex in examples))
    place_to_indices = {p: [i for i, ex in enumerate(examples) if ex["place"] == p] for p in unique_places}

    aurocs = []
    for _ in range(n_boot):
        sampled_places = rng.choice(unique_places, size=len(unique_places), replace=True)
        idx = []
        for p in sampled_places:
            idx.extend(place_to_indices[p])
        y_sample = target[idx]
        p_sample = probs[idx]
        if len(set(y_sample)) > 1:
            aurocs.append(roc_auc_score(y_sample, p_sample))
    return np.array(aurocs)
