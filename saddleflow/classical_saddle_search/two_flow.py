"""A maintained orientation q, not a recomputed minimum eigenvector."""

from .reflected_force import run_mode_flow


def run_two_flow(pes, x0, **options):
    return run_mode_flow(pes, x0, orientation="two_flow", **options)
