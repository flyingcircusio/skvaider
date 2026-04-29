import batou_ext.nix
from batou.component import Component
from batou.lib.file import File


@batou_ext.nix.rebuild
class SkvaiderSettings(Component):
    def configure(self):
        self.provide("skvaider-settings", self)
        self += File(
            "/etc/local/nixos/skvaider-source.nix",
            source="skvaider-source.nix",
        )
