from dataclasses import dataclass
from typing import Literal

type ModelMap = dict[str, set[str]]


@dataclass
class ModelSpec:
    id: str
    size: dict[str, int]
    instances: int


@dataclass
class BackendSlot:
    id: str
    capacity: dict[str, int]
    map_in: Literal["in", "out"]

    def __post_init__(self):
        self.available: dict[str, int] = dict(self.capacity)
        self.placed: dict[str, ModelSpec] = {}
        self.picked: set[str] = set()

    def fits(self, model: ModelSpec) -> bool:
        return all(
            self.available.get(r, 0) >= cost for r, cost in model.size.items()
        )

    def tightness(self, model: ModelSpec) -> int:
        """Remaining capacity after placing — lower is tighter (better fit)."""
        return sum(
            self.available.get(r, 0) - cost for r, cost in model.size.items()
        )

    def place(self, model: ModelSpec) -> None:
        self.placed[model.id] = model
        for r, cost in model.size.items():
            self.available[r] = self.available.get(r, 0) - cost

    def pick(self, model: ModelSpec) -> None:
        """Claim one instance of model. Places it first if not already pre-placed."""
        if model.id not in self.placed:
            self.place(model)
        self.picked.add(model.id)

    def consider_eviction(self, model: ModelSpec) -> tuple[set[str], int]:
        """Find non-picked models to evict to fit model.

        Returns (ids_to_evict, waste_after_placement).

        Evicts largest-first to minimise the number of evictions needed.

        Returning (set(), -1) is an indicator that we can't evict anything
        to make room for this model.

        """
        if model.id in self.placed:
            return set(), -1
        evictable = sorted(
            (m for m in self.placed.values() if m.id not in self.picked),
            key=lambda m: -sum(m.size.values()),
        )
        freed = dict(self.available)
        to_evict: set[str] = set()
        for m in evictable:
            if all(freed.get(r, 0) >= cost for r, cost in model.size.items()):
                break
            to_evict.add(m.id)
            for r, cost in m.size.items():
                freed[r] = freed.get(r, 0) + cost
        if not all(freed.get(r, 0) >= cost for r, cost in model.size.items()):
            return set(), -1
        waste = sum(freed.get(r, 0) - cost for r, cost in model.size.items())
        return to_evict, waste

    def evict(self, to_evict: set[str]) -> None:
        for m_id in to_evict:
            model = self.placed.pop(m_id)
            for r, cost in model.size.items():
                self.available[r] = self.available.get(r, 0) + cost

    def unpick(self, model_id: str) -> None:
        """Remove a picked model, restoring its capacity."""
        model = self.placed.pop(model_id)
        self.picked.discard(model_id)
        for r, cost in model.size.items():
            self.available[r] = self.available.get(r, 0) + cost


def build_picklist(models: list[ModelSpec]) -> list[ModelSpec]:
    """Create an ordered list of models for placement.

    The list can be used to iteratively "pick" models and place them
    into the map resulting in a fair distribution on a potentially limited
    cluster that isn't able to fit all.

    """
    sorted_models = sorted(
        models,
        key=lambda m: (-sum(m.size.values()), m.id),
    )
    remaining = {m.id: m.instances for m in sorted_models}
    picklist: list[ModelSpec] = []

    while remaining:
        for m in sorted_models:
            if m.id not in remaining:
                continue
            picklist.append(m)
            remaining[m.id] -= 1
            if remaining[m.id] == 0:
                del remaining[m.id]

    return picklist


def _spread_placements(
    in_backends: list[BackendSlot],
    pre_existing: set[tuple[str, str]],
) -> None:
    """Move newly placed models from overloaded to underloaded backends."""
    if len(in_backends) < 2:
        return
    while True:
        by_load = sorted(in_backends, key=lambda b: (len(b.picked), b.id))
        least = by_load[0]
        most = by_load[-1]
        if len(most.picked) - len(least.picked) <= 1:
            break
        moveable = sorted(
            m_id
            for m_id in most.picked
            if (most.id, m_id) not in pre_existing
            and m_id not in least.picked
            and least.fits(most.placed[m_id])
        )
        if not moveable:
            break
        model_id = moveable[0]
        model = most.placed[model_id]
        most.unpick(model_id)
        least.pick(model)


def placement_map(
    backends: list[BackendSlot],
    models: list[ModelSpec],
    last_map: ModelMap,
) -> ModelMap:
    in_backends = [b for b in backends if b.map_in == "in"]

    # Record usage of previous model placements for backends that are
    # still in use.
    model_by_id = {m.id: m for m in models}
    for b in in_backends:
        for model_id in set(last_map.get(b.id, set())) & set(model_by_id):
            b.place(model_by_id[model_id])

    pre_existing: set[tuple[str, str]] = {
        (b.id, m_id) for b in in_backends for m_id in b.placed
    }

    for model in build_picklist(models):
        # Gravity: claim an unpicked instance already placed via last_map.
        for b in in_backends:
            if model.id in b.placed and model.id not in b.picked:
                b.pick(model)
                break
        else:
            # Direct placement: tightest-fit backend with room.
            candidates = [
                b
                for b in in_backends
                if model.id not in b.placed and b.fits(model)
            ]
            candidates.sort(
                key=lambda b: (b.tightness(model), b.id),
            )
            if candidates:
                candidates[0].pick(model)
            else:
                # Eviction: backend where evicting non-picked models makes room.
                best_backend: BackendSlot | None = None
                best_evict: set[str] = set()
                best_waste = 0
                for b in in_backends:
                    to_evict, waste = b.consider_eviction(model)
                    if waste == -1:
                        continue
                    if best_backend is None or waste < best_waste:
                        best_backend = b
                        best_evict = to_evict
                        best_waste = waste
                if not best_backend:
                    # No way to place this model, just keep going.
                    continue
                assert best_evict
                best_backend.evict(best_evict)
                best_backend.pick(model)

    _spread_placements(in_backends, pre_existing)

    result_map: ModelMap = {b.id: set() for b in backends}
    for b in in_backends:
        result_map[b.id] = set(b.picked)
    return result_map
