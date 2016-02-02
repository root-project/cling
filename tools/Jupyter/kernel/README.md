# Cling Kernel

C++ Kernel for Jupyter with Cling.

Requires ipykernel ≥ 4.0

## Install

To install the kernel with sources in cling/src:

    cd cling/src/tools/Jupyter/kernel/
    pip install -e .
    # register the kernelspec:
    jupyter kernelspec install [--user] cling
