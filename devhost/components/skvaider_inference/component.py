import batou_ext.nix
from batou.component import Component
from batou.lib.file import File


@batou_ext.nix.rebuild
class SkvaiderInference(Component):
    def configure(self):
        self.require_one("skvaider-settings", host=self.host)
        self.provide("skvaider-inference", self)
        self += File(
            "/etc/local/nixos/skvaider-inference.nix",
            source="skvaider-inference.nix",
        )
