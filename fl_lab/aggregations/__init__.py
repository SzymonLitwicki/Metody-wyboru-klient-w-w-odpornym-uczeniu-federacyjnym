from fl_lab.aggregations.base import Aggregator
from fl_lab.aggregations.bulyan import BulyanAggregator
from fl_lab.aggregations.fedavg import FedAvgAggregator
from fl_lab.aggregations.fltrust import FLTrustAggregator
from fl_lab.aggregations.krum import KrumAggregator
from fl_lab.aggregations.trimmed_mean import TrimmedMeanAggregator

_REGISTRY: dict[str, type[Aggregator]] = {
    "fedavg": FedAvgAggregator,
    "krum": KrumAggregator,
    "trimmed_mean": TrimmedMeanAggregator,
    "bulyan": BulyanAggregator,
    "fltrust": FLTrustAggregator,
}


def get_aggregator(name: str) -> Aggregator:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown aggregator {name!r}. Expected one of: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
