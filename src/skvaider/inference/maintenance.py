"""Maintenance mode for an inference server.

The phase is derived from the maintenance request, the manifest received
from the proxy and the current model states, so it cannot get stuck: if the
proxy never reacts, the phase stays at ``requested`` and the caller's
timeout aborts the maintenance.

Phases: ``requested`` (proxy still places models here), ``draining``
(manifest emptied, models unloading), ``drained`` (no models left, safe to
stop), ``none`` (not in maintenance).
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skvaider.manifest import (
    MaintenancePhase,
    MaintenanceStatus,
    Serial,
)

if TYPE_CHECKING:
    from skvaider.inference.model import Model


@dataclass
class Maintenance:
    """State of the maintenance request of an inference server."""

    requested: bool = False
    since: datetime.datetime | None = None
    reason: str | None = None

    def request(self, reason: str | None = None) -> None:
        """Idempotent; keeps the original start time."""
        if reason is not None:
            self.reason = reason
        if self.requested:
            return
        self.requested = True
        self.since = datetime.datetime.now(datetime.UTC)

    def leave(self) -> None:
        """Leave maintenance. Idempotent."""
        self.requested = False
        self.since = None
        self.reason = None

    @staticmethod
    def pending(models: Iterable[Model]) -> list[str]:
        """Models that are not fully stopped and thus block a drain."""
        return sorted(
            model.config.id
            for model in models
            if model.process_status != "stopped"
        )

    def phase(
        self,
        manifest: set[str],
        manifest_serial: Serial,
        models: Iterable[Model],
    ) -> MaintenancePhase:
        if not self.requested:
            return "none"
        if manifest_serial == Serial.floor():
            return "requested"
        if manifest:
            return "requested"
        if self.pending(models):
            return "draining"
        return "drained"

    def status(
        self,
        manifest: set[str],
        manifest_serial: Serial,
        models: Iterable[Model],
    ) -> MaintenanceStatus:
        models = list(models)
        return MaintenanceStatus(
            phase=self.phase(manifest, manifest_serial, models),
            since=self.since,
            reason=self.reason,
            manifest=sorted(manifest),
            pending_models=self.pending(models),
        )
