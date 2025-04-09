---
layout: doc
title: The 101 GEMM Example
parent: Examples
nav_order: 1
has_children: false
---

TileFusion approaches the efficient implementation of a kernel by:

1. Managing dataflow over memory hierarchies.
2. Configuring tile primitives, such as tile shapes, layouts, and other parameters.

This is an example of a simple GEMM (General Matrix Multiplication) kernel written using TileFusion. For the complete example, please refer to [this directory](https://github.com/microsoft/TileFusion/blob/master/examples/01_gemm/01_gemm_global_reg/gemm.hpp).

## Configuration of the Tile Primitives

The core programming constructs in TileFusion are `Tile`, `TileLayout`, `TileIterator`, `Loader`, and `Storer`.

1. **Declare the `Tile`**: [GlobalTile](https://github.com/microsoft/TileFusion/blob/master/include/types/global.hpp) and [RegTile](https://github.com/microsoft/TileFusion/blob/master/include/types/register.hpp) are utilized to customize the shape and layout of 1D (vector) or 2D (matrix) arrays within the GPU's three memory hierarchies, known as a *Tile*.

2. **Declare the `TileIterator`**: Partition the `GlobalTile` into smaller, manageable sub-tiles for efficient processing.

3. **Declare Loader and Storer**: Loaders and Storers use cooperative threads to transfer a tile from the source to the target location. They operate at the CTA level and accept the following inputs:

   - **Warp Layout**
   - **Target Tile**
   - **Source Tile**

   Based on these parameters, they automatically infer a copy plan that partitions the data transfer work among the threads.

```cpp
using WarpLayout = RowMajor<2, 2>;

// operand A
using GlobalA = GlobalTile<InType, RowMajor<128, 256>>;
using IteratorA = TileIterator<GlobalA, TileShape<128, 32>>;
using RegA = RegTile<BaseTileRowMajor<__half>, RowMajor<8, 8>>;
using ALoader = GlobalToRegLoader<RegA, WarpLayout, kRowReuseCont>;

// operand B
using GlobalB = GlobalTile<InType, ColMajor<256, 64>>;
using IteratorB = TileIterator<GlobalB, TileShape<32, 64>>;
using RegB = RegTile<BaseTileColMajor<__half>, ColMajor<8, 4>>;
using BLoader = GlobalToRegLoader<RegB, WarpLayout, kColReuseCont>;

// output C
using GlobalC = GlobalTile<AccType, RowMajor<128, 64>>;
using RegC = RegTile<BaseTileRowMajor<float>, RowMajor<8, 8>>;
using CStorer = RegToGlobalStorer<GlobalC, RegC, WarpLayout>;
```

> **Note**: To simplify the demonstration, this example involves only two memory levels: global memory and registers. TileFusion also applies similar concepts to [SharedTile](https://github.com/microsoft/TileFusion/blob/master/include/types/shared.hpp).

## Dataflow Over Memory Hierarchies

The the kernel is defined as implementing the following dataflow over memory hierarchies:

```cpp
template <typename InType, typename AccType,
          typename IteratorA, typename RegA, typename LoaderA,
          typename IteratorB, typename RegB, typename LoaderB,
          typename GlobalC, typename RegC, typename CStorer>
__global__ void simple_gemm(const InType* dA, const InType* dB,
                            AccType* dC) {
    IteratorA gAs(dA);
    RegA rA;
    LoaderA loader_a;

    IteratorB gBs(dB);
    RegB rB;
    LoaderB loader_b;

    RegC acc;

    for (int k = 0; k < IteratorA::sc1; ++k) {
        loader_a(gAs(k), rA);
        loader_b(gBs(k), rB);
        __syncthreads();

        gemm(rA, rB, acc);
    }
    __syncthreads();

    GlobalC gC(dC);
    CStorer storer_c;
    storer_c(acc, gC);
}
```

The `TileIterator` (`IteratorA`, `IteratorB` in lines 6 and 10) serves as a syntactic interface for defining tile partitions. It is used to divide the `GlobalTile` into smaller sub-tiles and iterate over them.

`Loader` and `Storer` (declared in lines 8, 12, and 26) are efficient methods for loading and storing data, transferring data between memory hierarchies using specialized hardware-accelerated instructions (lines 17, 18, and 27). Tiles of data are cooperatively loaded into the `RegTile`, which is stored in each thread's local register file.

Once the data is loaded into a thread's local register file, `gemm` (in line 21) performs matrix multiplication using TensorCore's warp-level matrix multiply-and-accumulate (WMMA) instruction on the `BaseTile`s. The specialized data distribution required by TensorCore is automatically maintained by TileFusion's `RegTile` layout.

After the `gemm` operation is completed, the data in the `RegTile` is cooperatively stored back from registers to global memory using the `RegToGlobalStorer`.
