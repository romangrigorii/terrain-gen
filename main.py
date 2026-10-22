"""Pygame viewer.

    python main.py [--seed N] [--size WxH] [--windowed WxH]
"""
import argparse

import pygame

from terrain import Config
from viewer import Viewer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--size", default="640x448", help="map size, WxH")
    ap.add_argument("--windowed", default="1280x820", help="window size, WxH")
    ap.add_argument("--plates", type=int, default=None)
    a = ap.parse_args(argv)
    mw, mh = (int(v) for v in a.size.lower().split("x"))
    ww, wh = (int(v) for v in a.windowed.lower().split("x"))
    cfg = Config(width=mw, height=mh, seed=a.seed)
    if a.plates:
        cfg.n_plates = a.plates
    Viewer(cfg, (ww, wh)).run()
    pygame.quit()

if __name__ == "__main__":
    main()
