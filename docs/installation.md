---
layout: page
title: Installation
nav_order: 2
has_children: false
no_toc: true
---

TileFusion can be used as a lightweight C++ library with header-only usage, or it can be built as a Python library. You can choose to build either one.

### Prerequisites

TileFusion requires:

- C++20 host compiler
- CUDA 12.0 or later
- GCC version 10.0 or higher to support C++20 features

Download the repository:

```bash
git clone git@github.com:microsoft/TileFusion.git
cd TileFusion && git submodule update --init --recursive
```

### Building the C++ Library

To build the project using the provided `Makefile`, simply run:

```bash
make
```

To run a single C++ unit test:

```bash
make unit_test_cpp CPP_UT=test_gemm
```

### Building the Python Package

1. Build the wheel:

   ```bash
   python setup.py build bdist_wheel
   ```

2. Clean the build:

   ```bash
   python setup.py clean
   ```

3. Install the Python package in editable mode (recommended for development):

   ```bash
   python setup.py develop
   ```

   This allows you to edit the source code directly without needing to reinstall it repeatedly.

### Running Unit Tests

Before running the Python unit tests, you need to build and install the Python package (see the [Building the Python Package](#building-the-python-package) section).

- **Run a single Python unit test**:

  ```bash
  pytest tests/python/test_scatter_nd.py
  ```

- **Run all Python unit tests**:

  ```bash
  python setup.py pytests
  ```

- **Run all C++ unit tests**:

  ```bash
  python setup.py ctests
  ```
