import batou_ext.nix
from batou.component import Attribute, Component
from batou.lib.file import File


@batou_ext.nix.rebuild
class SkvaiderGateway(Component):
    token = Attribute(str, default="developer")
    inference_url = Attribute(str, default="")

    def configure(self):
        self.require_one("skvaider-settings", host=self.host)
        inference = self.require_one("skvaider-inference")
        if not self.inference_url:
            self.inference_url = f"http://{inference.host.name}:8000"
        self += File(
            "/etc/local/nixos/skvaider-gateway.nix",
            source="skvaider-gateway.nix",
        )
