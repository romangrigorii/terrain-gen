"""Trees: the one stage that reads the finished world and changes nothing in it."""
from dataclasses import dataclass

import numpy as np

from . import climate, grid, noise

CONIFER, BROADLEAF, PALM, SCRUB = range(4)
KIND_NAMES = ["conifer", "broadleaf", "palm", "scrub"]

KIND_BY_BIOME = np.array([
    SCRUB,      # ocean - never drawn, cover is zero there
    SCRUB,      # ice
    SCRUB,      # tundra
    CONIFER,    # taiga
    SCRUB,      # desert
    SCRUB,      # steppe
    BROADLEAF,  # temperate forest
    CONIFER,    # temperate rainforest
    SCRUB,      # shrubland
    PALM,       # savanna - flat-crowned, an acacia read at three pixels
    PALM,       # tropical seasonal forest
    BROADLEAF,  # tropical rainforest
], dtype=np.int8)


@dataclass
class Trees:
    cover: np.ndarray   # canopy fraction per cell, 0..1
    pos: np.ndarray     # (n, 2) float, (y, x) in cells
    kind: np.ndarray    # (n,) index into KIND_NAMES
    size: np.ndarray    # (n,) relative, around 1.0

    def __len__(self):
        return len(self.kind)


def canopy(world, cfg, rng):
    """Canopy fraction per cell. Four terms, every one of them continuous."""
    mi = climate.moisture_index(world.temp, world.swing, world.runoff, cfg)

    n = noise.fbm(*mi.shape, rng, periods=cfg.tree_grove_periods, octaves=4)
    mi = mi * (1.0 + cfg.tree_grove * n / max(1e-9, float(n.std())))

    t = np.clip((mi - cfg.tree_mi_open) /
                max(1e-9, cfg.tree_mi_closed - cfg.tree_mi_open), 0.0, 1.0)
    cover = t * t * (3.0 - 2.0 * t)                 # smoothstep

    warmest = world.temp + world.swing
    treeline = np.clip((warmest - climate.TREE_C) / cfg.tree_line_fade, 0.0, 1.0)
    cover *= treeline

    near = np.exp(-grid.edt(~(world.rivers | world.lakes)) / cfg.tree_gallery_reach)
    cover += cfg.tree_gallery * near * (1.0 - cover) * treeline

    cover = np.clip(cover, 0.0, 1.0)
    cover[~world.land] = 0.0
    cover[world.lakes] = 0.0
    cover[world.biome == climate.ICE] = 0.0
    return cover


def kind_mix(biome, land, cfg):
    """Probability of each kind per cell, as a blurred local composition."""
    k = KIND_BY_BIOME[biome]
    m = land.astype(float)
    p = np.stack([grid.blur((k == i) * m, cfg.tree_mix)
                  for i in range(len(KIND_NAMES))], axis=-1)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-9)


def scatter(cover, biome, land, cfg, rng):
    """Turn the density field into individual trees, on a jittered grid."""
    h, w = cover.shape
    s = max(0.5, cfg.tree_spacing)
    gy, gx = np.meshgrid(np.arange(0, h, s), np.arange(0, w, s), indexing="ij")
    y = gy.ravel() + rng.random(gy.size) * s
    x = gx.ravel() + rng.random(gx.size) * s
    yi = np.clip(y.astype(int), 0, h - 1)
    xi = x.astype(int) % w
    p = cover[yi, xi]
    keep = rng.random(p.size) < p
    yi, xi = yi[keep], xi[keep]
    size = (0.55 + 0.45 * cover[yi, xi]) * np.exp(
        rng.normal(0.0, cfg.tree_size_var, yi.size))
    c = np.cumsum(kind_mix(biome, land, cfg)[yi, xi], axis=1)
    u = rng.random(yi.size)
    kind = np.minimum((u[:, None] > c).sum(axis=1), len(KIND_NAMES) - 1)
    return Trees(cover=cover, pos=np.stack([y[keep], x[keep]], axis=1),
                 kind=kind.astype(np.int8), size=size)


def build(world, rng):
    return scatter(canopy(world, world.cfg, rng), world.biome, world.land,
                   world.cfg, rng)
