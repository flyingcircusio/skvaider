import datetime

from skvaider.proxy.map import DOWN_OUT_GRACE, BackendMap, In


def test_maintenance_takes_the_backend_out_and_back_in():
    m = BackendMap()
    m.up.mark("up")
    assert m.is_in

    for phase in ("requested", "draining", "drained"):
        m.maintenance.mark(phase)
        assert not m.is_in
        assert m.in_maintenance

    m.maintenance.mark("none")
    assert m.is_in


def test_probe_failure_keeps_placement_until_the_grace_expires():
    m = BackendMap()
    m.up.mark("up")

    m.up.mark("down")
    # A flapping probe must not churn placements.
    assert m.is_in

    m.up.restore(
        "down", datetime.datetime.now(datetime.UTC) - 2 * DOWN_OUT_GRACE
    )
    m.tick()
    assert not m.is_in


def test_probe_failure_announces_the_state_but_not_a_placement_change():
    m = BackendMap()
    m.up.mark("up")
    placements: list[In] = []
    m.subscribe(lambda: placements.append(m.in_))

    m.up.mark("down")
    assert placements == ["in"]

    m.up.restore(
        "down", datetime.datetime.now(datetime.UTC) - 2 * DOWN_OUT_GRACE
    )
    m.tick()
    assert placements == ["in", "out"]


def test_maintenance_phases_do_not_move_the_placement():
    m = BackendMap()
    m.up.mark("up")
    placements: list[In] = []
    m.subscribe(lambda: placements.append(m.in_))

    m.maintenance.mark("requested")
    m.maintenance.mark("draining")
    m.maintenance.mark("drained")

    assert placements == ["out", "out", "out"]


def test_unchanged_marks_do_not_notify():
    m = BackendMap()
    m.up.mark("up")
    calls: list[int] = []
    m.subscribe(lambda: calls.append(1))

    assert not m.up.mark("up")
    assert not m.maintenance.mark("none")
    assert calls == []


def test_restore_is_silent_and_recomputes_placement():
    m = BackendMap()
    m.up.mark("up")
    calls: list[int] = []
    m.subscribe(lambda: calls.append(1))

    other = BackendMap()
    other.up.mark("down")
    other.maintenance.mark("drained")

    m.restore(other.to_record())

    assert calls == []
    assert m.maintenance.state == "drained"
    assert not m.is_in


def test_record_roundtrip():
    m = BackendMap()
    m.up.mark("up")
    m.maintenance.mark("draining")

    restored = BackendMap()
    restored.restore(m.to_record())

    assert restored.up.state == "up"
    assert restored.maintenance.state == "draining"
    assert restored.up.last_change == m.up.last_change
    assert not restored.is_in
