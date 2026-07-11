import numpy as np
from torch.utils.data import Dataset, Subset


def _get_labels(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    return np.asarray([int(dataset[i][1]) for i in range(len(dataset))])


def iid_partition(dataset: Dataset, n_clients: int, seed: int) -> list[Subset]:
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = np.arange(n)
    rng.shuffle(indices)
    splits = np.array_split(indices, n_clients)
    return [Subset(dataset, s.tolist()) for s in splits]


def dirichlet_partition(
    dataset: Dataset,
    n_clients: int,
    alpha: float,
    seed: int,
) -> list[Subset]:
    rng = np.random.default_rng(seed)
    labels = _get_labels(dataset)
    n_classes = int(labels.max()) + 1

    client_indices: list[list[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        cutoffs = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, cutoffs)
        for ci, cs in zip(client_indices, class_splits):
            ci.extend(int(x) for x in cs)

    return [Subset(dataset, idx) for idx in client_indices]


def pathological_partition(
    dataset: Dataset,
    n_clients: int,
    shards_per_client: int,
    seed: int,
) -> list[Subset]:
    rng = np.random.default_rng(seed)
    labels = _get_labels(dataset)
    n_total_shards = n_clients * shards_per_client
    if n_total_shards > len(dataset):
        raise ValueError(
            f"n_clients * shards_per_client ({n_total_shards}) "
            f"exceeds dataset size ({len(dataset)})"
        )

    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, n_total_shards)

    shard_ids = np.arange(n_total_shards)
    rng.shuffle(shard_ids)

    client_indices: list[list[int]] = []
    for i in range(n_clients):
        ids = shard_ids[i * shards_per_client : (i + 1) * shards_per_client]
        client_indices.append(np.concatenate([shards[j] for j in ids]).astype(int).tolist())

    return [Subset(dataset, idx) for idx in client_indices]
