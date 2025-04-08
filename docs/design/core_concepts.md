---
layout: mathjax
title: Core Programming Concepts
---

## Types

### Tile

### TileLayout

<div align="center">
  <img src="../../assets/images/matrix_layout.png" width="200"/><br>
  Fig: The matrix layout used for the global and shared memory tiles.
</div>

<div align="center">
  <img src="../../assets/images/tiled_matrix_layout.png" width="200"/><br>
  Fig: The tiled matrix layout used for the shared memory tile.
</div>

<div align="center">
  <img src="../../assets/images/register_tile_layout.png" width="500"/><br>
  Fig: The TensorCore register tile layout.
</div>

### TileIterator

<div align="center">
  <img src="../../assets/images/tile_iterator.png" width="400"/><br>
  Fig: Partition tensor using a tile iterator.
</div>

## Tile Transfer with Loaders and Storers

<div align="center">
  <img src="../../assets/images/loader_and_storer.png" width="600"/><br>
  Fig: A tile is transferred between memory hierarchies using a loader and a storer.
</div>
