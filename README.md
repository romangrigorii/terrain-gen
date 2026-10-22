# Terrain generator

Procedural worlds from plate tectonics. Plates raise the mountains, rain and rivers carve them, climate picks the biomes, and trees grow last. The map wraps east to west and closes with ocean at the poles.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
python gui.py
python edit_gui.py
python preview.py 7
python -m terrain --preset alpine
python test_terrain.py
python -m pytest tests
```

`main.py` takes `--seed N` (default 7), `--size WxH` (default 640x448), `--windowed WxH`, and `--plates N`.

Drag, WASD, or the arrows pan. The wheel zooms. `Z` fits the map. `1`-`0` and `-` select the first eleven layers, `[` and `]` step through the rest. `R` builds a new seed, `T` rebuilds the current one, `N` types a seed. `G` shows plate motion, `P` saves the current layer, `E` exports, `I` imports a `config.json`. Esc or `Q` quits.

Sliders in the Edit menu rebuild when you release them: plates, map size, land/sea, shelf roughness, rivers, lakes, and temperature.

`gui.py` is the asset studio. It keeps every generated tile in one atlas, paints mountains and valleys across tile edges, and loads that surface in 3D (`D`: left-drag pans, right-drag orbits, wheel zooms). Presets are `earthlike`, `archipelago`, `supercontinent`, `alpine`, `arid`, and `volcanic`. Export writes the atlas, with optional hierarchical detail, under `out/`.

`edit_gui.py` opens one of those exports and embellishes it: sculpt the height, blend a palette of grounds (soil, sand, rock, snow, ice, mud, water, and volcanic), cut paths and ponds, and place trees, rocks, and buildings. Paint weights sum to one, so strokes mix instead of replacing the whole cell. It writes back into that folder. The first save keeps the untouched rasters in `authored/original`. From the studio, Embellish maps opens the newest export. A tiled export is edited one tile at a time.

## Layout

Settings are `terrain/config.py`. `terrain/world.py` runs the stages and returns a `World`. The viewer is `viewer.py`.

| Stage | Module |
|---|---|
| plates | `terrain/tectonics.py` |
| relief, hotspots, texture, coastline | `terrain/elevation.py` |
| erosion | `terrain/hydrology.py` |
| rain, temperature, biomes | `terrain/climate.py` |
| lakes and rivers | `terrain/rivers.py` |
| trees | `terrain/trees.py` |
| pictures | `terrain/render.py` |
| engine export | `terrain/export.py` |

```python
from terrain import generate
world = generate(seed=12, n_plates=9, land_fraction=0.4)
```

`world.height` is elevation with sea level at 0. Beside it: `surface`, `flow`, `runoff`, `rivers`, `lakes`, `temp`, `biome`, `tect`, `water`, and `trees`.

Pixel-sized settings are tuned at `ref_width` and scaled to the map. `generate(export.load_config(path))` rebuilds an export from its seed and settings.

Export writes `out/export/seed<N>/`: `height16.png`, `lakes16.png`, `biome.png`, `canopy.png`, `rivers.json`, `config.json`. Each PNG has a JSON sidecar. A later export of the same seed uses `seed<N>-2` rather than overwriting.

## License

MIT. See [LICENSE](LICENSE).
