"""State containers for BIRCH leaf-link rewiring helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BirchLeafLinkPlan:
    """Role-level rewiring plan for a BIRCH leaf split."""

    left_prev_role: str | None
    left_next_role: str
    right_prev_role: str
    right_next_role: str | None
    prev_next_role: str | None
    next_prev_role: str | None
