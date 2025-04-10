---
layout: home
nav_order: 1
---

<div class="home-header">
  <div align="center">
    <img src="assets/images/logos/TileFusion-logo.png" width="120"/>
  </div>
</div>

<h1>TileFusion: A High-Level, Modular<br>Tile Processing Library</h1>

<div class="home-shields" align="center">
  <a href="https://github.com/microsoft/TileFusion" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-TileFusion-blue.svg" alt="GitHub">
  </a>
</div>

**[TileFusion](https://github.com/microsoft/TileFusion)**, derived from the research presented in this [paper](https://dl.acm.org/doi/pdf/10.1145/3694715.3695961), is an efficient C++ macro kernel library designed to elevate the level of abstraction in CUDA C for tile processing. The library offers:

- **Higher-Level Programming Constructs**: TileFusion supports tiles across the three-level GPU memory hierarchy, providing device kernels for transferring tiles between CUDA memory hierarchies and for tile computation.
- **Modularity**: TileFusion enables applications to process larger tiles built out of BaseTiles in both time and space, abstracting away low-level hardware details.
- **Efficiency**: The library's BaseTiles are designed to match TensorCore instruction shapes and encapsulate hardware-specific performance parameters, ensuring optimal utilization of TensorCore capabilities.

A core design goal of **TileFusion** is to allow users to understand and utilize provided primitives using logical concepts, without delving into low-level hardware complexities. The library rigorously separates data flow across the memory hierarchy from the configuration of individual macro kernels. This design choice enables performance enhancements through tuning, which operates in three possible ways:

- **Structural Tuning**: Designs various data flows while keeping kernel configurations unchanged.
- **Parameterized Tuning**: Adjusts kernel configurations while maintaining the same data flow.
- **Combined Tuning**: Integrates both structural and parameterized tuning approaches simultaneously.

In summary, **TileFusion** encourages algorithm developers to focus on designing the data flow of their algorithms using efficient tile primitives. It can be utilized as:

1. A lightweight C++ library with header-only usage, offering superior readability, modifiability, and debuggability.
2. A Python library with pre-existing kernels bound to PyTorch.
