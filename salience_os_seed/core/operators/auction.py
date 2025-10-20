"""Compute auction used by the salience controller.

Operators submit bids representing the expected marginal return of executing
that operator. The controller augments its bandit score with the auction
adjustment to prioritise high-return, low-cost actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from ..controller.actions import ControllerAction


@dataclass(frozen=True)
class AuctionBid:
    """Represents a single operator bid."""

    action: ControllerAction
    expected_gain: float
    expected_cost: float

    @property
    def score(self) -> float:
        return self.expected_gain - self.expected_cost


class ComputeAuction:
    """Collect and score compute bids per decision step."""

    def __init__(self) -> None:
        self._bids: Dict[str, AuctionBid] = {}

    def submit(self, bid: AuctionBid) -> None:
        self._bids[self._key(bid.action)] = bid

    def resolve(self) -> Mapping[ControllerAction, float]:
        return {bid.action: bid.score for bid in self._bids.values()}

    def clear(self) -> None:
        self._bids.clear()

    @staticmethod
    def _key(action: ControllerAction) -> str:
        return f"depth={action.cot_depth}|op={action.operator.name}|patch={action.patch.name}"
