from __future__ import annotations


class LMTrainer:
    """No-op placeholder trainer for language-model workflows."""

    def train(self, out_dir: str) -> None:
        _ = out_dir
