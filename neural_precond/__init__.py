"""Neural preconditioner package."""

__all__ = ["PolyPrecondSolver"]


def __getattr__(name):
    if name == "PolyPrecondSolver":
        from neural_precond.solver import PolyPrecondSolver
        return PolyPrecondSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
