# tta/tta_registry.py
from tta.methods.no_tta import NoTTA
from tta.methods.bn_adapt import BNAdapt
from tta.methods.tent import Tent
from tta.methods.eata import EATA
from tta.methods.sar import SAR
from tta.methods.cotta import CoTTA
from tta.methods.rotta import RoTTA
from tta.methods.t3a import T3A
from tta.methods.deyo import DeYO
from tta.methods.lame import LAME
from tta.methods.roid import ROID
from tta.methods.sotta import SoTTA
from tta.methods.vida import ViDA
from tta.methods.come import COME
from tta.methods.rem import REM
from tta.methods.sicl import SICL
from tta.methods.rmemsafe import RMemSafe

TTA_REGISTRY = {
    "no_tta":   (NoTTA,   {"lr": 0.0}),
    "bn_adapt": (BNAdapt, {"lr": 0.0}),
    "tent":     (Tent,    {"lr": 1e-3}),
    "eata":     (EATA,    {"lr": 1e-3}),
    "sar":      (SAR,     {"lr": 1e-3}),
    "cotta":    (CoTTA,   {"lr": 1e-3}),
    "rotta":    (RoTTA,   {"lr": 1e-3}),
    "t3a":      (T3A,     {"lr": 0.0}),
    "deyo":     (DeYO,    {"lr": 1e-3}),
    # 2022-2025 SOTA additions
    "lame":     (LAME,    {"lr": 0.0}),
    "roid":     (ROID,    {"lr": 1e-3}),
    "sotta":    (SoTTA,   {"lr": 1e-3}),
    "vida":     (ViDA,    {"lr": 1e-3}),
    # 2025-2026 SOTA additions
    "come":     (COME,    {"lr": 1e-3}),
    "rem":      (REM,     {"lr": 1e-3}),
    "sicl":     (SICL,    {"lr": 0.0}),
    "rmemsafe": (RMemSafe, {"lr": 1e-3}),
}


def get_tta(name, model, device='cuda', num_classes=2):
    cls, kwargs = TTA_REGISTRY[name]
    kwargs = dict(kwargs)
    if name in ("sotta", "t3a"):        # methods whose init needs the class count
        kwargs["num_classes"] = num_classes
    return cls(model, device=device, **kwargs)
