"""Checkpoint selection shared by the generative adaptation trainers."""

import math

SELECTION_METRIC = "mean_clean_artefact_f1"


def validation_f1(metrics):
    """Weight clean and pooled artefact F1 equally, irrespective of view counts."""
    clean = float(metrics["clean"]["f1"])
    artefact = float(metrics["weak"]["f1"])
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (clean, artefact)):
        raise ValueError("Checkpoint selection requires finite clean and artefact F1 in [0, 1]")
    return (clean + artefact) / 2


def validate_selection_resume(checkpoint):
    if checkpoint.get("selection_metric") != SELECTION_METRIC:
        raise ValueError(
            "Checkpoint uses an older or unknown selection criterion. "
            "Start a new run with a new save_dir/tag instead of mixing criteria."
        )
