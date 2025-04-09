---
layout: doc
title: Program Concepts in TileFusion
---

TileFusion operates on four core concepts: Tile, Layout, TileIterator, and Loader/Storer, which facilitate the transfer of tiles between memory hierarchies.

## Types

### Tile

Represents the fundamental unit of data. Variants such as GlobalTile, SharedTile, and RegTile are utilized to customize the shape and layout of 1D (vector) or 2D (matrix) arrays located in GPU's three memory hierarchies.

### Tile Layout

Tile Layout is a parameterized function that maps an integer tuple to an integer. It is utilized for organizing data, threads, and warps. In TileFusion, there are three types of layouts, and **they are composable**. However, it is important to note that arbitrary nested composability is not supported; composition can be performed only once.

- **Matrix Layout**: Characterized by shape and strides, this layout has two specializations: Row-major and Column-major. It is used for the efficient access of global and shared memory tiles.
- **Tiled Matrix Layout**: Facilitates the efficient access of shared memory tiles or describes the thread or warp layout.
- **Register Tile Layout**: Designed for the efficient access of TensorCore register tiles.

Figure 1 to 3 show how these three layouts works.

<div align="center">
  <img src="../../assets/images/matrix_layout.png" width="200"/><br>
  Fig 1: The matrix layout used for the global and shared memory tiles.
</div>

Shown in Figure 1, Matrix Layout is characterized by shape and strides. This layout has two specializations: Row-major and Column-major. It is used for the efficient access of global and shared memory tiles.

<div align="center">
  <img src="../../assets/images/tiled_matrix_layout.png" width="200"/><br>
  Fig 2: The tiled matrix layout used for the shared memory tile.
</div>

Shown in Figure 2, Tiled matrix layout can be understood as a matrix layout composed with another matrix layout. This layout facilitates the efficient access of shared memory tiles or describes the thread or warp layout.

<div align="center">
  <img src="../../assets/images/register_tile_layout.png" width="500"/><br>
  Fig 3: The TensorCore register tile layout.
</div>

Shown in Figure 3, the `BaseTileRowMajor` and `BaseTileColumnMajor` are specialized layouts designed for Tensor Core MMA operations. These register layouts combine a BaseTileMatrixLayout with a matrix layout. The rule here is that **arbitrary nested composability isn't supported; composition can only be performed once**.

### TileIterator

TileIterator provides syntactic interfaces for defining tile partitions, facilitating the systematic traversal of tiles. It has two variants: `GTileIterator` and `STileIterator`.

```cpp
using GlobalA = GlobalTile<InType, tl::RowMajor<kTM, kK, kK>>;
using GIteratorA = GTileIterator<GlobalA, TileShape<kTM, kTK>>;

using SharedA = SharedTile<InType, tl::RowMajor<kTM, kTK>>;
using SIteratorA = STileIterator<SharedA, TileShape<kTM, kRK>>;
```

<div align="center">
  <img src="../../assets/images/tile_iterator.png" width="400"/><br>
  Fig 4: Partition tensor using a tile iterator.
</div>

Given that a tile represents a larger data region, the tile shape specifies the dimensions of a smaller tile. The TileIterator then divides the larger tile into smaller tiles along each dimension.

## Tile Transfer with Loader and Storer

<div align="center">
  <img src="../../assets/images/loader_and_storer.png" width="600"/><br>
  Fig 5: A tile is transferred between memory hierarchies using a loader and a storer.
</div>

Loaders and Storers use cooperative threads to transfer a tile from the source to the target location. They operate at the CTA level and accept the following inputs: Warp Layout, Target Tile, and Source Tile. Based on these parameters, they automatically infer a copy plan that partitions the data transfer work among the threads.
