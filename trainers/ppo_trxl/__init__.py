__all__ = ["PPOTRxLTrainer"]


def __getattr__(name: str):
    if name == "PPOTRxLTrainer":
        from .trainer import PPOTRxLTrainer

        return PPOTRxLTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
