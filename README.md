Cling - The Interactive C++ Interpreter
=========================================

The main repository is at [https://github.com/root-project/cling](https://github.com/root-project/cling)


Overview
--------
Cling is an interactive C++ interpreter, built on top of Clang and LLVM compiler
infrastructure. Cling implements the [read-eval-print loop
(REPL)](http://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop)
concept, in order to leverage rapid application development. Implemented as a
small extension to LLVM and Clang, the interpreter reuses their strengths such
as the praised concise and expressive compiler diagnostics.

See also [cling's web page.](https://rawcdn.githack.com/root-project/cling/master/www/index.html)

Please note that some of the resources are rather old and most of the stated
limitations are outdated.
  * [talks](www/docs/talks)
  * http://blog.coldflake.com/posts/2012-08-09-On-the-fly-C++.html
  * http://solarianprogrammer.com/2012/08/14/cling-cpp-11-interpreter/
  * https://www.youtube.com/watch?v=f9Xfh8pv3Fs
  * https://www.youtube.com/watch?v=BrjV1ZgYbbA
  * https://www.youtube.com/watch?v=wZZdDhf2wDw
  * https://www.youtube.com/watch?v=eoIuqLNvzFs


Installation
------------
### Release Notes
See our [release notes](docs/ReleaseNotes.md) to find what's new.


### Binaries
Our nightly binary snapshots are currently unavailable.


### Building from Source

See also the instructions [on the webpage](https://root.cern/cling/cling_build_instructions/).

#### Building Cling as a Standalone Project

If Clang and LLVM (cling-latest version) are not installed, you need to build them first:

```bash
git clone https://github.com/root-project/llvm-project.git
cd llvm-project
git checkout cling-latest
cd ..
mkdir llvm-build && cd llvm-build
cmake -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX" -DCMAKE_BUILD_TYPE=Release ../llvm-project/llvm
cmake --build .
```

Once Clang and LLVM (cling-latest version) are installed, you can build Cling. If they are already installed, you can skip the previous step and proceed with the following:

> Note: Ensure you are outside the llvm-project and llvm-build directories before proceeding, as LLVM, Clang, and Cling do not allow building inside the source directory.

```bash
git clone https://github.com/root-project/cling.git
mkdir cling-build && cd cling-build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="The root of your LLVM build directory" -DLLVM_DIR="The directory containing LLVM's CMake modules" ../cling
cmake --build .
```

Example CMake command:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=../llvm-build -DLLVM_DIR=../llvm-build/lib/cmake/llvm ../cling
```

#### Building Cling Along with LLVM (Recommended)
If Clang and LLVM are not installed, you can build them together with Cling:

```bash
git clone https://github.com/root-project/llvm-project.git
cd llvm-project
git checkout cling-latest
cd ..
git clone https://github.com/root-project/cling.git
mkdir cling-build && cd cling-build
cmake -DLLVM_EXTERNAL_PROJECTS=cling -DLLVM_EXTERNAL_CLING_SOURCE_DIR=../cling/ -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX" -DCMAKE_BUILD_TYPE=Release ../llvm-project/llvm
cmake --build . --target clang cling
```

#### Jupyter notebooks

To enable support for cling in Jupyter notebooks, after building cling, run:

```bash
cmake --build . --target libclingJupyter
```

Usage
-----
Assuming we're in the build folder.

If Cling is built as a standalone project, you need to specify the include directory for headers:

```bash
./bin/cling -I"../cling/include" '#include <stdio.h>' 'printf("Hello World!\n");'
```

If build Cling as part of LLVM:

```bash
./bin/cling '#include <stdio.h>' 'printf("Hello World!\n");'
```

To get started run:
```bash
./bin/cling --help
```
or
```bash
./bin/cling
[cling]$ .help
```


Jupyter
-------
Cling comes with a [Jupyter](http://jupyter.org) kernel. After building cling,
install Jupyter and cling's kernel by following the README.md in
[tools/Jupyter](tools/Jupyter). Make sure cling is in your PATH when you start jupyter!


Citing Cling
------------
```latex
% Peer-Reviewed Publication
%
% 19th International Conference on Computing in High Energy and Nuclear Physics (CHEP)
% 21-25 May, 2012, New York, USA
%
@inproceedings{Cling,
  author = {Vassilev,V. and Canal,Ph. and Naumann,A. and Moneta,L. and Russo,P.},
  title = {{Cling} -- The New Interactive Interpreter for {ROOT} 6}},
  journal = {Journal of Physics: Conference Series},
  year = 2012,
  month = {dec},
  volume = {396},
  number = {5},
  pages = {052071},
  doi = {10.1088/1742-6596/396/5/052071},
  url = {https://iopscience.iop.org/article/10.1088/1742-6596/396/5/052071/pdf},
  publisher = {{IOP} Publishing}
}
```

Developers' Corner
==================
[Cling's latest doxygen documentation](http://cling.web.cern.ch/cling/doxygen/)


Contributions
-------------
Every contribution is considered a donation and its copyright and any other
related rights become exclusive ownership of the person who merged the code or
in any other case the main developers of the "Cling Project".

We warmly welcome external contributions to the Cling! By providing code,
you agree to transfer your copyright on the code to the "Cling project".
Of course you will be duly credited and your name will appear on the
contributors page, the release notes, and in the [CREDITS file](CREDITS.txt)
shipped with every binary and source distribution. The copyright transfer is
necessary for us to be able to effectively defend the project in case of
litigation.


License
-------
Please see our [LICENSE](LICENSE.TXT).


Releases
--------
Our release steps to follow when cutting a new release:
  1. Update [release notes](docs/ReleaseNotes.md)
  2. Remove `~dev` suffix from [VERSION](VERSION)
  3. Add a new entry in the news section of our [website](www/news.html)
  4. Commit the changes.
  5. `git tag -a v0.x -m "Tagging release v0.x"`
  6. Tag `cling-patches` of `clang.git`:
     `git tag -a cling-v0.x -m "Tagging clang for cling v0.x"`
  7. Create a draft release in github and copy the contents of the release notes.
  8. Wait for green builds.
  9. Upload binaries to github (Travis should do this automatically).
  10. Publish the tag and announce it on the mailing list.
  11. Increment the current version and append `~dev`.
