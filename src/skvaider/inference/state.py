import os
from pathlib import Path

from skvaider.inference.manager import ModelManager

models_dir = Path(os.environ.get("MODELS_DIR", "models"))
manager = ModelManager(models_dir=models_dir)
