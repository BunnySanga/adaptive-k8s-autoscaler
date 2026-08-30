from __future__ import annotations

import math
import os

from locust import FastHttpUser, LoadTestShape, between, task


class WorkloadUser(FastHttpUser):
    wait_time = between(0.01, 0.2)

    @task
    def cpu_work(self) -> None:
        self.client.get("/work?duration_ms=80&complexity=250", name="/work")


class SixPatternShape(LoadTestShape):
    """Select a workload with LOAD_PATTERN.

    Supported values:
    stable, gradual_increase, sudden_burst, repeated_short_spikes,
    sustained_growth, spike_then_drop.
    """

    pattern = os.getenv("LOAD_PATTERN", "stable")
    run_time = int(os.getenv("LOAD_DURATION_SECONDS", "1800"))
    spawn_rate = float(os.getenv("LOCUST_SPAWN_RATE", "10"))

    def tick(self) -> tuple[int, float] | None:
        elapsed = self.get_run_time()
        if elapsed > self.run_time:
            return None
        users = self.users_for(elapsed)
        return max(1, int(users)), self.spawn_rate

    def users_for(self, elapsed: float) -> float:
        if self.pattern == "stable":
            return 35

        if self.pattern == "gradual_increase":
            return 10 + 90 * (elapsed / self.run_time)

        if self.pattern == "sudden_burst":
            return 25 if elapsed < 420 else 140

        if self.pattern == "repeated_short_spikes":
            cycle = elapsed % 240
            return 120 if 60 <= cycle <= 95 else 25

        if self.pattern == "sustained_growth":
            return 15 + 120 / (1 + math.exp(-(elapsed - self.run_time / 2) / 140))

        if self.pattern == "spike_then_drop":
            if 360 <= elapsed <= 540:
                return 150
            return 30

        return 35

