from __future__ import annotations

import importlib.util
import sys
from config import TEST

def _model(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TEST / filename)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    assert spec.loader is not None; spec.loader.exec_module(module)
    return module.AdvancedLatticeCNN

SingleChannelCNN = _model("ablation_model_1c", "model.py")
SoftMaskCNN = _model("ablation_model_6c", "model_v3.py")
