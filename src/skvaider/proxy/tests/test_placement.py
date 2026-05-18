from typing import Literal

from skvaider.proxy.placement import (
    BackendSlot,
    ModelSpec,
    build_picklist,
    placement_map,
)


def B(id: str, ram: int, map_in: Literal["in", "out"] = "in") -> BackendSlot:
    return BackendSlot(id=id, capacity={"ram": ram}, map_in=map_in)


def M(id: str, ram: int, instances: int) -> ModelSpec:
    return ModelSpec(id=id, size={"ram": ram}, instances=instances)


def ids(picklist: list[ModelSpec]) -> list[str]:
    return [m.id for m in picklist]


# --- build_picklist ---


def test_picklist_single_instance():
    assert ids(build_picklist([M("a", 4, 1)])) == ["a"]


def test_picklist_multiple_instances():
    assert ids(build_picklist([M("a", 4, 3)])) == ["a", "a", "a"]


def test_picklist_larger_model_first():
    assert ids(build_picklist([M("small", 1, 1), M("large", 8, 1)])) == [
        "large",
        "small",
    ]


def test_picklist_round_robin_respects_desired_count():
    # large wants 2, small wants 1 — large should appear twice, small once
    assert ids(build_picklist([M("large", 8, 2), M("small", 1, 1)])) == [
        "large",
        "small",
        "large",
    ]


def test_picklist_total_length():
    assert len(build_picklist([M("a", 8, 2), M("b", 4, 2), M("c", 1, 4)])) == 8


def test_picklist_guaranteed_round_first():
    # Every model should appear in the first |models| slots
    first_round = ids(
        build_picklist([M("a", 8, 2), M("b", 4, 2), M("c", 1, 4)])
    )[:3]
    assert set(first_round) == {"a", "b", "c"}


def test_picklist_stable_tiebreak():
    # Equal size: alphabetical by id
    assert ids(build_picklist([M("z", 4, 1), M("a", 4, 1)])) == ["a", "z"]
    assert ids(build_picklist([M("a", 4, 1), M("z", 4, 1)])) == ["a", "z"]


# --- placement_map: basics ---


def test_empty_inputs():
    assert placement_map([], [], {}) == {}


def test_no_backends():
    map = placement_map(
        [],
        [
            M("m", 4, 1),
        ],
        {},
    )

    assert map == {}


def test_no_models():
    result = placement_map(
        [
            B("b1", 10),
        ],
        [],
        {},
    )
    assert result == {"b1": set()}


def test_simple_placement():
    result = placement_map(
        [
            B("b1", 10),
        ],
        [
            M("m", 5, 1),
        ],
        {},
    )
    assert result == {"b1": {"m"}}


def test_out_backend_excluded():
    result = placement_map(
        [
            B("b1", 10, map_in="out"),
        ],
        [
            M("m", 5, 1),
        ],
        {},
    )
    assert result == {"b1": set()}


def test_capacity_respected():
    result = placement_map(
        [
            B("b1", 4),
        ],
        [
            M("m", 8, 1),
        ],
        {},
    )
    assert result == {"b1": set()}


def test_model_not_placed_twice_on_same_backend():
    # Even with 3 desired instances, a model can only appear once per backend
    result = placement_map(
        [
            B("b1", 100),
        ],
        [
            M("m", 1, 3),
        ],
        {},
    )
    assert result == {"b1": {"m"}}


def test_multiple_models_distributed():
    # a and b both need 2 instances but only fit once per backend (size 8, cap 10)
    # a placed on b1, b placed on b2; second instances can't fit anywhere
    result = placement_map(
        [
            B("b1", 10),
            B("b2", 10),
        ],
        [
            M("a", 8, 2),
            M("b", 8, 2),
        ],
        {},
    )
    assert result == {
        "b1": {"a"},
        "b2": {"b"},
    }


# --- specified test case ---


def test_specified_scenario():
    """4 backends (size 10), 2 offline; all models must still get one instance."""
    # Picklist: [L, M, S, L, M, S, S, S]
    # L→b1(avail=2), M→b2(avail=6), S→b1(avail=1, tightest),
    # L×2→no room, M×2→no room, S×2→b2, S×3,×4→both full
    result = placement_map(
        [
            B("b1", 10),
            B("b2", 10),
            B("b3", 10, map_in="out"),
            B("b4", 10, map_in="out"),
        ],
        [
            M("L", 8, 2),
            M("M", 4, 2),
            M("S", 1, 4),
        ],
        {},
    )
    assert result == {
        "b1": {"L", "S"},
        "b2": {"M", "S"},
        "b3": set(),
        "b4": set(),
    }


# --- gravity ---


def test_gravity_retains_existing_placement():
    # m was on b2 in last_map → stays on b2; b1 stays empty
    assert placement_map(
        [
            B("b1", 10),
            B("b2", 10),
        ],
        [
            M("m", 5, 1),
        ],
        {"b2": {"m"}},
    ) == {
        "b1": set(),
        "b2": {"m"},
    }


def test_gravity_drops_out_backend():
    # b2 is out → m migrates to b1; b2 appears empty
    result = placement_map(
        [
            B("b1", 10),
            B("b2", 10, map_in="out"),
        ],
        [
            M("m", 5, 1),
        ],
        {"b2": {"m"}},
    )
    assert result == {
        "b1": {"m"},
        "b2": set(),
    }


def test_gravity_strips_unknown_model():
    # "ghost" is no longer in model config → dropped from placement
    result = placement_map(
        [B("b1", 10)],
        [M("m", 5, 1)],
        {"b1": {"m", "ghost"}},
    )
    assert result == {"b1": {"m"}}


def test_gravity_new_backend_starts_empty():
    # b2 absent from last_map → starts empty; m stays on b1
    result = placement_map(
        [
            B("b1", 10),
            B("b2", 10),
        ],
        [
            M("m", 5, 1),
        ],
        {"b1": {"m"}},
    )
    assert result == {
        "b1": {"m"},
        "b2": set(),
    }


# --- eviction ---


def test_eviction_makes_room():
    # b1: cap=10. last_map pre-loads tiny(1) → avail=9.
    # Picklist: [large(9), tiny(1)]. large fits directly (9≥9); tiny claimed via gravity.
    assert placement_map(
        [B("b1", 10)],
        [M("large", 9, 1), M("tiny", 1, 1)],
        {"b1": {"tiny"}},
    ) == {"b1": {"large", "tiny"}}


def test_eviction_needed_when_capacity_full():
    # b1: cap=10. last_map pre-loads c(5) → avail=5.
    # Picklist: [big(8), c(5)]. big needs 8 > 5 → evicts unpicked c (avail→10).
    # c cannot be placed afterwards (avail=2 < 5).
    assert placement_map(
        [B("b1", 10)],
        [M("big", 8, 1), M("c", 5, 1)],
        {"b1": {"c"}},
    ) == {"b1": {"big"}}


# --- spreading ---


def test_spreading_balances_load():
    # 3 backends (400), m1(100, 2 instances) + m2(200, 2 instances), cold start.
    # Picklist places all 4 instances on b1+b2. Spreading moves one model to b3.
    assert placement_map(
        [B("b1", 400), B("b2", 400), B("b3", 400)],
        [M("m1", 100, 2), M("m2", 200, 2)],
        {},
    ) == {
        "b1": {"m1", "m2"},
        "b2": {"m2"},
        "b3": {"m1"},
    }


def test_spreading_respects_gravity():
    # Gravity pins m to b1; b2 keeps its model; no unwanted moves.
    assert placement_map(
        [B("b1", 100), B("b2", 100)],
        [M("m", 50, 1)],
        {"b1": {"m"}},
    ) == {
        "b1": {"m"},
        "b2": set(),
    }


def test_eviction_tightest_fit_preferred():
    # b1: cap=11, pre-loads pre1(8) → avail=3.
    # b2: cap=9,  pre-loads pre2(3) → avail=6.
    # Picklist: [big(8), pre1(8), pre2(3)].
    # big needs eviction: b1 would free 11 (waste=3), b2 would free 9 (waste=1) → b2 chosen.
    # pre1 claimed via gravity on b1; pre2 then fits on b1 (avail=3).
    assert placement_map(
        [
            B("b1", 11),
            B("b2", 9),
        ],
        [
            M("big", 8, 1),
            M("pre1", 8, 1),
            M("pre2", 3, 1),
        ],
        {"b1": {"pre1"}, "b2": {"pre2"}},
    ) == {
        "b1": {"pre1", "pre2"},
        "b2": {"big"},
    }
