Applications
------------


1. **C++ in Jupyter Notebook - Xeus Cling:**

The `Jupyter Notebook <https://jupyter.org/>`_ technology allows users to create
and share documents that contain live code, equations, visualizations and
narrative text. It enables data scientists to easily exchange ideas or
collaborate by sharing their analyses in a straight-forward and reproducible
way. Jupyter’s official C++ kernel(`Xeus-Cling
<https://github.com/jupyter-xeus/xeus-cling>`_) relies on Xeus, a C++
implementation of the kernel protocol, and Cling. Using C++ in the Jupyter
environment yields a different experience to C++ users. For example, Jupyter’s
visualization system can be used to render rich content such as images,
therefore bringing more interactivity into the Jupyter’s world. You can find
more information on `Xeus Cling's Read the Docs
<https://xeus-cling.readthedocs.io/en/latest/>`_ webpage.


2. **Interactive CUDA C++ with Cling:**

`CUDA <https://blogs.nvidia.com/blog/2012/09/10/what-is-cuda-2/>`_ is a platform
and Application Programming Interface (API) created by `NVIDIA
<https://www.nvidia.com/en-us/>`_.  It controls `GPU
<https://en.wikipedia.org/wiki/Graphics_processing_unit>`_ (Graphical Processing
Unit) for parallel programming, enabling developers to harness the power of
graphic processing units (GPUs) to speed up applications. As an example,
`PIConGPU <https://github.com/ComputationalRadiationPhysics/picongpu>`_ is a
CUDA-based plasma physics application to solve the dynamics of a plasma by
computing the motion of electrons and ions in the plasma field.  Interactive GPU
programming was made possible by extending Cling functionality to compile CUDA
C++ code. The new Cling-CUDA C++ can be used on Jupyter Notebook platform, and
enables big, interactive simulation with GPUs, easy GPU development and
debugging, and effective GPU programming learning.


3. **Clad:**

`Clad <https://compiler-research.org/clad/>`_ enables automatic differentiation
(AD) for C++. It was first developed as a plugin for Cling, and is now a plugin
for Clang compiler. Clad is based on source code transformation. Given C++
source code of a mathematical function, it can automatically generate C++ code
for computing derivatives of the function. It supports both forward-mode and
reverse-mode AD.

4. **Cling for live coding music and musical instruments:**

The artistic live coding community has been growing steadily since around the
year 2000. The Temporary Organisation for the Permanence of Live Art Programming
(TOPLAP) has been around since 2004, Algorave (algorithmic rave parties)
recently celebrated its tenth birthday, and six editions of the International
Conference on Live Coding (ICLC) have been held. A great many live coding
systems have been developed during this time, many of them exhibiting exotic and
culturally specific features that professional software developers are mostly
unaware of. In this framework, Cling has been used as the basis for a C++ based
live coding synthesiser (`TinySpec-Cling
<https://github.com/nwoeanhinnogaehr/tinyspec-cling>`_). In another example,
Cling has been installed on a BeagleBoard to bring live coding to the Bela
interactive audio platform (`Using the Cling C++ Interpreter on the Bela
Platform
<https://gist.github.com/jarmitage/6e411ae8746c04d6ecbee1cbc1ebdcd4>`_). These
two examples show the potential mutual benefits for increased engagement between
the Cling community and the artistic live coding community.

5. **Clion:** The `CLion <https://www.jetbrains.com/clion/>`_ platform is a
Integrating Development Environment (`IDE
<https://en.wikipedia.org/wiki/Integrated_development_environment>`_) for C and
C++ by `JetBrains <https://www.jetbrains.com/>`_. It was developed with the aim
to enhance developer's productivity with a smart editor, code quality assurance,
automated refactorings and deep integration with the CMake build system. CLion
integrates Cling, which can be found by clicking on Tool. Cling enables
prototyping and learning C++ in CLion. You can find more information on `CLion's
building instructions
<https://www.jetbrains.com/help/clion/cling-integration.html#install-cling>`_.


