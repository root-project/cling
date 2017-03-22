### Example project using cling as library

This example project uses cling as an external library.
It compiles code and calls it, moving values from the compiled part to the
interpreted part and back.

It showcases how to use cling as a library, and shows how to set up a simple
CMake configuration that uses cling.


### How to build

After installing cling (say into /where/cling/is/installed), configure this
project using CMake like this:
```bash
cmake -Dcling_DIR=/cling-install-dir/lib/cmake/cling /cling-source-dir/tools/cling/tools/demo
make && ./cling-demo
```
