from .agent import TRPOAgent

__all__ = ["TRPOAgent", "TRPOTrainer"]


def __getattr__(name: str):
    if name == "TRPOTrainer":
        from .trainer import TRPOTrainer

        return TRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
