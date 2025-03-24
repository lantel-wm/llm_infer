# llm_infer

A lightweight C++ inference framework for Large Language Models (LLMs) with both CPU and GPU support.

## Overview

llm_infer is a toy project designed to provide efficient inference capabilities for LLMs. It implements key components found in modern transformer-based architectures with a focus on performance and flexibility.

## Features

- Multi-backend support (CPU, CUDA)
- Tensor operations with GPU acceleration
- Common transformer layers implementation:
  - Linear layers
  - Multi-Head Attention (MHA)
  - RMSNorm
  - Rotary Position Embedding (RoPE)
  - SwiGLU activation
  - Vector addition
  - Embedding layers
- Memory management for efficient resource utilization
- Core data types and tensor operations

## Project Structure

- **src/**: Source code
  - **core/**: Core functionality (memory management, configuration, types, status)
  - **tensor/**: Tensor operations and manipulation
  - **layer/**: Neural network layer implementations
  - **kernel/**: Low-level kernel operations for CPU and GPU
- **test/**: Unit tests
- **python_test/**: Python notebooks and scripts for testing and validation

## Dependencies

- CUDA Toolkit
- LAPACK and BLAS
- Armadillo (linear algebra library)
- gflags (command line flags)
- glog (logging)

## Building

The project uses CMake for building:

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

This framework can be used to implement inference for transformer-based LLMs by composing the provided layers.

## Development Status

This is a toy project for educational purposes and is under active development. New features and optimizations are being added regularly.