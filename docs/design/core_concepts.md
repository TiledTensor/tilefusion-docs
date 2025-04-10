---
layout: mathjax
title: Program Concepts in TileFusion
---

TileFusion operates on four core concepts: `Tile`, `Layout`, `TileIterator`, and `Loader/Storer`, which facilitate the transfer of tiles between memory hierarchies.

## Types

Core types in TileFusion are defined in the [types](https://github.com/microsoft/TileFusion/tree/master/include/types) directory.

### Tile

A tile is a 1D (vector) or 2D (matrix) array that resides within one of the three GPU memory hierarchies. A tile is typically characterized by three attributes:

- **Shape**: The dimensions of the tile, specified by the number of elements along each axis.
- **Layout**: Layout is a parameterized function that maps a tuple of integer coordinates (representing the elements in the tile) to an integer. The lexicographical order of these coordinates can determine the sequence of elements within the tile.
- **ElementType**: The data type of the elements stored in the tile.

Based on the memory hierarchy where a tile resides, there are three different variants: [GlobalTile](https://github.com/microsoft/TileFusion/blob/master/include/types/global.hpp), [SharedTile](https://github.com/microsoft/TileFusion/blob/master/include/types/shared.hpp), and [RegTile](https://github.com/microsoft/TileFusion/blob/master/include/types/register.hpp).

#### Global Memory Tile

A 2D tile in global memory with a shape of $[64, 64]$, a `RowMajor` layout, and a `float` element type can be defined as follows:

```cpp
using Global = GlobalTile<float, RowMajor<64, 64>>;
```

#### Shared Memory Tile

To define an equivalent tile located in shared memory:

```cpp
// `is_swizzled = true` indicates the tile is swizzled in shared memory,
// which is a common practice to enhance shared memory access performance.
// The default value is false, which simplifies debugging.
using Shared = SharedTile<float, RowMajor<64, 64>, is_swizzled=true>;
```

<p class="highlight-note"><span class="note-prefix">Note:</span> Both Global and Shared memory tiles use a RowMajor layout, although their physical memory layouts differ. This difference will be explained in the next section, <a href="https://tiledtensor.github.io/tilefusion-docs/docs/design/core_concepts#tiled-matrix-layout">Tiled Matrix Layout</a>. Users don't need to concern themselves with these details. They only need to know that a shared memory tile is a 2D array with dimensions $[64, 64]$ and a RowMajor layout. The tile primitive will manage the layout automatically.</p>

#### Register File Tile

For tiles located in the register file, the definition differs slightly. In CUDA, registers are thread-local. Consequently, when the aforementioned tile is located in the register file, it is partitioned across threads in the CTA. Therefore, the register tile held by an individual thread is defined as follows:

```cpp
using Reg = RegTile<BaseTileRowMajor<float>, RowMajor<4, 4>>;
```

We will further discuss the second parameter of `RegTile` in the next section: <a href="https://tiledtensor.github.io/tilefusion-docs/docs/design/core_concepts#register-tile-layout">Register Tile Layout</a>.

### Tile Layout

The shape of a tile defines a high-dimensional space, with each coordinate in this space represented by an integer tuple. The layout of a tile is a function mapping this integer tuple to an integer, offering a comprehensive and logical description of data, threads, warps, and other resources. In TileFusion, there are conceptually three types of layouts: **Matrix Layout**, **Tiled Matrix Layout**, and **Register Tile Layout**.

<p class="highlight-note"><span class="note-prefix">Note:</span> These three layouts are inter-composable, but an important simplification we made is that arbitrary nested composability is not supported; composition can be performed only once. This will be explained in the examples below.</p>

#### Matrix Layout

The [matrix layout](https://github.com/microsoft/TileFusion/blob/master/include/types/layout.hpp#L48) is defined by its shape and strides. This layout is utilized for global and shared memory tiles, as well as for specifying the numbering of threads or warps. It is declared as follows:

```cpp
using Layout = MatrixLayout<64 /*Rows*/, 64 /*Columns*/,
                            64 /*Row Stride*/, 1 /*Column Stride*/>;

// usage:
// layout is a callable function that maps a tuple of integers to an integer
Layout layout;
for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
        int idx = layout(i, j);  // idx = i * row_stride + j * column_stride
    }
}
```

This is equivalent to:

```cpp
using Layout = RowMajor<64, 64>;
```

As illustrated in Figure 1, the default element order of the above matrix layout follows the conventional row-major format.

<div align="center">
  <img src="../../assets/images/matrix_layout.png" width="200"/><br>
  Fig 1: The row-major matrix layout.
</div>
<br>

Similarly, the column-major matrix layout is defined as:

```cpp
using Layout = MatrixLayout<64 /*Rows*/, 64 /*Columns*/,
                            1 /*Row Stride*/, 64 /*Column Stride*/>;
```

This is equivalent to:

```cpp
using Layout = ColMajor<64, 64>;
```

Row-major and column-major layouts are two specializations of the matrix layout.

#### Tiled Matrix Layout

The <span class="text-red">tiled matrix layout is specifically designed for the efficient access of shared memory tiles</span>, and can be understood as a matrix layout composed with another matrix layout in concept.

A shared memory tile with a shape of `[64, 64]` and a `RowMajor` has a tiled matrix layout internally.

```cpp
using Shared = SharedTile<float, RowMajor<64, 64>, is_swizzled=true>;
```

For the shared memory tile mentioned above, Figure 2 demonstrates how data is stored in the tiled matrix layout.

<div align="center">
  <img src="../../assets/images/tiled_matrix_layout.png" width="250"/><br>
  Fig 2: The tiled matrix layout used for the shared memory tile.
</div>
<br>

In a tiled matrix layout, the inner matrix is stored in a contiguous block of memory. The outer layout treats each inner matrix as a single element, arranging these elements into another matrix layout.

Specifically, let's revisit the `RowMajor<64, 64>` for shared memory tile declared in the section [Shared Memory Tile](#shared-memory-tile). It represents a matrix layout that is comprised of another matrix layout. For a detailed explanation of the rationale behind this approach, please refer to [Tiles in Shared Memory](https://tiledtensor.github.io/tilefusion-docs/docs/design/tiles_in_shared_memory).

<p class="highlight-note"><span class="note-prefix">Note:</span> Specifically, let's revisit the RowMajor<64, 64> layout for the shared memory tile, as declared in the section <a href="https://tiledtensor.github.io/tilefusion-docs/docs/design/core_concepts#shared-memory-tile">Shared Memory Tile</a>. This layout represents a matrix that is composed of another matrix layout. For a detailed explanation of the rationale behind this approach, please refer to <a href="https://tiledtensor.github.io/tilefusion-docs/docs/design/tiles_in_shared_memory">Tiles in Shared Memory</a>.</p>

#### Register Tile Layout

The <span class="text-red">rThe register tile layout is specifically designed to efficiently feed data to TensorCore</span>. Conceptually similar to the tiled matrix layout, it is a depth-two nested array with a `BaseTileMatrixLayout` as the inner layout and a `MatrixLayout` as the outer layout.

Specifically, let's revisit the register tile, as declared in the section [Register File Tile](#register-file-tile). TensorCore's MMA instruction has a hardware-prescribed tile shape and layout for the input operands. We prescribe a $[16, 16]$ basic building block to effectively leverage the MMA instruction. As shown on the left of Figure 3, a $16 \times 16$ basic tile feeding into the TensorCore is cooperatively held by a single warp. The first thread in the warp holds data in four segments, as indicated by the colors, and so on with the other threads in the warp. For a thread's register tile, `BaseTileRowMajor` and `BaseTileColumnMajor` store these four segments in the single thread's local register file.

<div align="center">
  <img src="../../assets/images/register_tile_layout.png" width="500"/><br>
  Fig 3: The TensorCore register tile layout.
</div>
<br>

Register layouts, based on `BaseTileRowMajor` and `BaseTileColumnMajor`, can be understood as a depth-two nested array with a `BaseTileMatrixLayout` as the inner layout and a `MatrixLayout` as the outer layout. The register tile in Figure 3's right is equivalent to the following definition:

```cpp
using Reg = RegTile<BaseTileRowMajor<float>, RowMajor<2, 3>>;
```

<p class="highlight-note"><span class="note-prefix">Note:</span>The interface for the register tile is coupled with the declaration of `RegisterTile`. The interface for specifying the register tile layout will be refined in the future to align more clearly with the underlying concept. For now, users can safely assume that implementations are guaranteed to follow the above description.</p>

### GlobalTileIterator and SharedTileIterator

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

## Loader and Storer for Tiles

<div align="center">
  <img src="../../assets/images/loader_and_storer.png" width="600"/><br>
  Fig 5: A tile is transferred between memory hierarchies using a loader and a storer.
</div>

Loaders and Storers use cooperative threads to transfer a tile from the source to the target location. They operate at the CTA level and accept the following inputs: Warp Layout, Target Tile, and Source Tile. Based on these parameters, they automatically infer a copy plan that partitions the data transfer work among the threads.
