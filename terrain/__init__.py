"""Procedural worlds from plates, rain, rivers, and climate."""
from .tiles import Tile, TileSpec, generate_tile
from .world import Config, World, generate

__all__ = ["Config", "World", "generate", "Tile", "TileSpec", "generate_tile"]
