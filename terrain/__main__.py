"""CLI: python -m terrain --preset archipelago --width 512 --height 384 --advanced"""
import argparse

from .presets import PRESETS, make_config
from .world import generate
from . import export


def main():
    ap = argparse.ArgumentParser(description="Reproducible procedural world builder")
    ap.add_argument("--preset", choices=tuple(PRESETS), default="earthlike")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--erosion", choices=("stream_power", "thermal", "none"), default="stream_power")
    ap.add_argument("--depressions", choices=("iterative", "priority_flood"), default="iterative")
    ap.add_argument("--wind", choices=("belts", "easterly", "westerly"), default="belts")
    ap.add_argument("--noise-backend", choices=("numpy", "numba"), default="numpy")
    ap.add_argument("--out", default="out/export")
    ap.add_argument("--advanced", action="store_true")
    a = ap.parse_args()
    cfg = make_config(a.preset, width=a.width, height=a.height, seed=a.seed,
                      erosion_model=a.erosion, depression_method=a.depressions,
                      wind_mode=a.wind, noise_backend=a.noise_backend)
    w = generate(cfg)
    path = export.bundle(w, parent=a.out, advanced=a.advanced)
    print(f"Exported: {path}")

if __name__ == "__main__":
    main()
