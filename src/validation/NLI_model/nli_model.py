"""Cross-encoder NLI model wrapper with GPU support.

Wraps ``sentence_transformers.CrossEncoder`` for both pretrained inference
and fine-tuned checkpoint loading.  Supports single-GPU, multi-GPU
(DataParallel), and CPU fallback.
"""

from __future__ import annotations

from typing import Any

# Lazy imports — sentence-transformers pulls in heavy torch deps
_CROSSENCODER = None


def _get_crossencoder():
    global _CROSSENCODER
    if _CROSSENCODER is None:
        from sentence_transformers import CrossEncoder
        _CROSSENCODER = CrossEncoder
    return _CROSSENCODER


def _get_torch():
    import torch
    return torch


class NLIModel:
    """NLI cross-encoder for entailment/contradiction/neutral scoring.

    Parameters
    ----------
    model_path:
        HuggingFace model ID (e.g. ``cross-encoder/nli-deberta-v3-small``)
        or local checkpoint path from fine-tuning.
    device:
        ``"cuda:0"``, ``"cpu"``, or ``"auto"`` (auto-detect GPU).
    max_length:
        Maximum token length for premise+hypothesis pairs.
    """

    _LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
    _LABEL_ORDER = ["contradiction", "entailment", "neutral"]

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        torch = _get_torch()
        CrossEncoder = _get_crossencoder()

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = CrossEncoder(
            model_path,
            device=torch.device(device),
            max_length=max_length,
        )
        self.device = device

    def predict(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Run NLI inference on premise/hypothesis pairs.

        Returns a list of dicts with ``label``, ``confidence``, and
        ``scores`` (contradiction/entailment/neutral).
        """
        torch = _get_torch()
        pairs = list(zip(premises, hypotheses))

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=True,
        )
        # scores: (n, 3) tensor → [contradiction, entailment, neutral]
        probs = scores.softmax(dim=1)
        labels = scores.argmax(dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()
        raw = scores.cpu().numpy()

        results: list[dict[str, Any]] = []
        for i in range(len(premises)):
            results.append({
                "label": self._LABEL_MAP[int(labels[i])],
                "confidence": round(float(confs[i]), 4),
                "scores": {
                    "contradiction": round(float(raw[i][0]), 4),
                    "entailment":    round(float(raw[i][1]), 4),
                    "neutral":       round(float(raw[i][2]), 4),
                },
            })
        return results

    @staticmethod
    def map_to_verdict(label: str) -> str:
        """Map NLI label to judge-style verdict."""
        return {
            "entailment": "entailed",
            "neutral": "uncertain",
            "contradiction": "not_entailed",
        }.get(label, "uncertain")
