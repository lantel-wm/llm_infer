#!/bin/bash

# Unset PYTHONPATH and CONDA_PREFIX to avoid Miniconda environment interference
unset PYTHONPATH
unset CONDA_PREFIX

# Set LD_LIBRARY_PATH to use system libraries first
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Run the test
$(dirname "$0")/test_memory_manager "$@" 