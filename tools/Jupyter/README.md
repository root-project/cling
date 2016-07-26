# Cling Kernel

C++ Kernel for Jupyter with Cling.

Requires ipykernel ≥ 4.0

## Install

To install the kernel with sources in src/tools/cling:

    export PATH=/cling-install-prefix/bin:$PATH
    cd /cling-install-prefix/share/cling/Jupyter/kernel

    pip install -e .
    # or: pip3 install -e .

    # register the kernelspec:
    jupyter-kernelspec install [--user] cling
    # or: jupyter kernelspec install [--user] cling

To run it:

    jupyter-notebook
    # or: jupyter notebook
