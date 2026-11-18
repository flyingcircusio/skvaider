"""Placement membership of a backend in the proxy's model map."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Literal

import structlog
from pydantic import BaseModel

from skvaider.manifest import MaintenancePhase

log = structlog.stdlib.get_logger()

type Up = Literal["up", "down"]
type In = Literal["in", "out"]

DOWN_OUT_GRACE = datetime.timedelta(minutes=15)


class MapRecord(BaseModel):
    """Persisted map state of one backend."""

    up: Up
    up_last_change: datetime.datetime
    maintenance: MaintenancePhase
    maintenance_last_change: datetime.datetime


class Flag[T]:
    """A typed state value that remembers when it changed.

    Subscribers are notified synchronously on every real transition, so they
    must not block and must not mark another flag: derivation belongs to the
    owner of the flag, not to its observers.
    """

    def __init__(self, state: T) -> None:
        self.state: T = state
        self.last_change = datetime.datetime.now(datetime.UTC)
        self._subscribers: list[Callable[[T], None]] = []

    def mark(self, state: T) -> bool:
        """Set the state; notify subscribers iff it changed."""
        if state == self.state:
            return False
        self.state = state
        self.last_change = datetime.datetime.now(datetime.UTC)
        for subscriber in list(self._subscribers):
            try:
                subscriber(state)
            except Exception:
                log.exception("map subscriber failed")
        return True

    def restore(self, state: T, last_change: datetime.datetime) -> None:
        """Set the state without notifying (state file load)."""
        self.state = state
        self.last_change = last_change

    def subscribe(self, subscriber: Callable[[T], None]) -> None:
        self._subscribers.append(subscriber)

    @property
    def age(self) -> datetime.timedelta:
        return datetime.datetime.now(datetime.UTC) - self.last_change


class BackendMap:
    """Placement membership of a backend: ``up``/``down`` and ``in``/``out``.

    Inputs are the probe result (``up``) and the maintenance phase the server
    reports; ``in_`` is derived from both. The valid states:

        state             up     maintenance        in_   notes
        ----------------  -----  -----------------  ----  -----------------
        up                up     none               in    normal
        up-maintenance    up     requested,         out   keeps serving the
                                 draining, drained        models it has
        down-grace        down   none, within       in    probe flapping,
                                 grace                    still placed
        down              down   none, past grace   out   gone, replaced
        down-maintenance  down   requested,         out
                                 draining, drained

    Transitions (``in_`` flips are the ones that change placement):

        from                  to                  trigger
        --------------------  ------------------  -------------------------
        down-grace, down      up                  probe succeeds (-> in)
        up, down-grace        up-maintenance      phase != "none" (-> out)
        down                  down-maintenance    phase != "none" (-)
        up-maintenance        up                  phase "none" (-> in)
        down-maintenance      down-grace, down    phase "none" (-> in / -)
        up                    down-grace          probe fails (-)
        up-maintenance        down-maintenance    probe fails (-)
        down-grace            down                grace expires, tick (-> out)

    A phase change inside an active maintenance (requested -> draining ->
    drained) does not move ``in_``. Time enters only through the grace
    period, which ``tick()`` evaluates: the ``up``/``maintenance``
    subscribers call it, and the monitor calls it once per cycle to catch
    the clock-driven transition.

    Subscribers fire on every real state change; the proxy persists the map
    and recomputes placements there.
    """

    up: Flag[Up]
    maintenance: Flag[MaintenancePhase]

    def __init__(self) -> None:
        self.up = Flag[Up]("down")
        self.maintenance = Flag[MaintenancePhase]("none")
        self._reported: In = self.in_
        self._subscribers: list[Callable[[], None]] = []
        self.up.subscribe(lambda _: self._input_changed())
        self.maintenance.subscribe(lambda _: self._input_changed())

    @property
    def in_(self) -> In:
        """Placement eligibility; computed, never stored."""
        if self.maintenance.state != "none":
            return "out"
        if self.up.state == "up" or self.up.age <= DOWN_OUT_GRACE:
            return "in"
        return "out"

    @property
    def is_up(self) -> bool:
        return self.up.state == "up"

    @property
    def is_in(self) -> bool:
        return self.in_ == "in"

    @property
    def in_maintenance(self) -> bool:
        return self.maintenance.state != "none"

    def tick(self) -> None:
        """Recompute; announces the grace expiry, harmless otherwise."""
        if self.in_ == self._reported:
            return
        self._reported = self.in_
        self._notify()

    def subscribe(self, subscriber: Callable[[], None]) -> None:
        self._subscribers.append(subscriber)

    def restore(self, record: MapRecord) -> None:
        """Load persisted state without announcing transitions."""
        self.up.restore(record.up, record.up_last_change)
        self.maintenance.restore(
            record.maintenance, record.maintenance_last_change
        )
        self._reported = self.in_

    def to_record(self) -> MapRecord:
        return MapRecord(
            up=self.up.state,
            up_last_change=self.up.last_change,
            maintenance=self.maintenance.state,
            maintenance_last_change=self.maintenance.last_change,
        )

    def _input_changed(self) -> None:
        self._reported = self.in_
        self._notify()

    def _notify(self) -> None:
        for subscriber in list(self._subscribers):
            try:
                subscriber()
            except Exception:
                log.exception("map subscriber failed")
