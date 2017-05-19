# Cling Kernel

C++ Kernel for Jupyter with Cling.

Requires ipykernel ≥ 4.0

## Install

To install the kernel with sources in src/tools/cling:

    export PATH=/cling-install-prefix/bin:$PATH
    cd /cling-install-prefix/share/cling/Jupyter/kernel

    pip install -e .
    # or: pip3 install -e .

    # register the kernelspec for C++17/C++14/C++11:
    # the user can install whichever kernel(s) they
    # wish:
    jupyter-kernelspec install [--user] cling-c++17
    jupyter-kernelspec install [--user] cling-c++14
    jupyter-kernelspec install [--user] cling-c++11

To run it:

    jupyter-notebook
    # or: jupyter notebook
