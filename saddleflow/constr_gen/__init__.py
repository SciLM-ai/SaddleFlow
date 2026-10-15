"""Optional UMA-based constraint-generation helpers for SaddleFlow.

The package contains two orthogonal utilities:

* inference-time force guidance, which can retarget a sampled terminal state;
* a training-time algebraic endpoint force constraint, which is opt-in and
  never changes the normal flow-matching objective unless configured.
"""

from .uma_force_guidance import (
    estimate_hf_direction,
    force_rms_and_max,
    guide_terminal_estimates,
    uma_forces,
)
from .training_force_constraint import (
    algebraic_endpoint_force_loss,
    free_coordinate_mask,
)

__all__ = [
    "estimate_hf_direction",
    "force_rms_and_max",
    "guide_terminal_estimates",
    "uma_forces",
    "algebraic_endpoint_force_loss",
    "free_coordinate_mask",
]
