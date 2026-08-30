from __future__ import annotations

import collections
from typing import Any


class StabilityFilter:

    def __init__(
        self,
        scale_up_consecutive: int = 3,
        scale_down_m: int = 5,
        scale_down_n: int = 7,
        burst_multiplier: float = 2.0,
        service_rate_per_pod: float = 30.0,
    ) -> None:
        self.scale_up_consecutive = scale_up_consecutive
        self.scale_down_m = scale_down_m
        self.scale_down_n = scale_down_n
        self.burst_multiplier = burst_multiplier
        self.service_rate_per_pod = service_rate_per_pod
        self._history: collections.deque[int] = collections.deque(maxlen=scale_down_n)

    def filter(
        self,
        proposed_replicas: int,
        current_replicas: int,
        decision_qps: float | None = None,
    ) -> tuple[int, dict[str, Any]]:
        self._history.append(proposed_replicas)

        if proposed_replicas == current_replicas:
            return current_replicas, {
                "stability_action": "no_change",
                "direction": "none",
                "history_len": len(self._history),
            }

        direction = "up" if proposed_replicas > current_replicas else "down"

        if (
            direction == "up"
            and decision_qps is not None
            and decision_qps > self.burst_multiplier * current_replicas * self.service_rate_per_pod
        ):
            return proposed_replicas, {
                "stability_action": "burst_bypass",
                "direction": direction,
                "decision_qps": decision_qps,
                "capacity": current_replicas * self.service_rate_per_pod,
                "history_len": len(self._history),
            }

        if direction == "up":
            if len(self._history) < self.scale_up_consecutive:
                return proposed_replicas, {
                    "stability_action": "allowed",
                    "direction": direction,
                    "reason": "cold_start",
                    "history_len": len(self._history),
                }
            recent = list(self._history)[-self.scale_up_consecutive :]
            all_up = all(r > current_replicas for r in recent)
            if all_up:
                return proposed_replicas, {
                    "stability_action": "allowed",
                    "direction": direction,
                    "consecutive_up": self.scale_up_consecutive,
                    "history_len": len(self._history),
                }
            return current_replicas, {
                "stability_action": "suppressed",
                "direction": direction,
                "proposed_replicas": proposed_replicas,
                "consecutive_up": sum(1 for r in recent if r > current_replicas),
                "required": self.scale_up_consecutive,
                "history_len": len(self._history),
            }

        if len(self._history) < self.scale_down_n:
            return proposed_replicas, {
                "stability_action": "allowed",
                "direction": direction,
                "reason": "cold_start",
                "history_len": len(self._history),
            }
        recent_n = list(self._history)[-self.scale_down_n :]
        down_count = sum(1 for r in recent_n if r < current_replicas)
        if down_count >= self.scale_down_m:
            return proposed_replicas, {
                "stability_action": "allowed",
                "direction": direction,
                "down_count": down_count,
                "required_m": self.scale_down_m,
                "window_n": self.scale_down_n,
                "history_len": len(self._history),
            }
        return current_replicas, {
            "stability_action": "suppressed",
            "direction": direction,
            "proposed_replicas": proposed_replicas,
            "down_count": down_count,
            "required_m": self.scale_down_m,
            "window_n": self.scale_down_n,
            "history_len": len(self._history),
        }
