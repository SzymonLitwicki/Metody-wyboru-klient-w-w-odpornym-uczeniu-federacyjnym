from dataclasses import dataclass, field


@dataclass
class RoundMetrics:
    round_num: int
    test_accuracy: float
    selected_clients: list[int]
    train_loss: float = 0.0
    test_loss: float = 0.0
    attack_detected: bool = False
    selection_time_ms: float = 0.0
    avg_train_time_ms: float = 0.0
    bytes_per_round: int = 0
    n_byzantine_in_cohort: int = 0
    byzantine_in_cohort_ids: list[int] = field(default_factory=list)
    selector_state_bytes: int = 0
    selection_peak_mem_bytes: int = 0


class MetricsCollector:
    def __init__(self) -> None:
        self.history: list[RoundMetrics] = []

    def record(self, metrics: RoundMetrics) -> None:
        self.history.append(metrics)

    def final_accuracy(self) -> float:
        if not self.history:
            return 0.0
        return self.history[-1].test_accuracy

    def rounds_to_accuracy(self, target: float) -> int | None:
        for m in self.history:
            if m.test_accuracy >= target:
                return m.round_num
        return None

    def total_selection_time_ms(self) -> float:
        return sum(m.selection_time_ms for m in self.history)
