Cling
=======================================

**Cling** is an interactive C++ interpreter built on top of `Clang <https://clang.llvm.org/>`_ and `LLVM <https://llvm.org/>`_.
It uses **LLVM**'s *Just-In-Time* (`JIT <https://en.wikipedia.org/wiki/Just-in-time_compilation>`_) compiler to provide a fast and optimized compilation pipeline. Cling uses the `read-eval-print-loop <https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop>`_ (**REPL**) approach, making rapid application development in C++ possible, avoiding the classic edit-compile-run-debug cycle approach. 
You can download Cling from `GitHub <https://github.com/root-project/cling>`_.


.. note::

  This project is under active development.
  Cling has its documentation hosted on Read the Docs.
   
   
.. figure:: images/sphynx_cat.jpeg
   



When and why was Cling developed?
--------------------------------------------
**Cling** is a core component of `ROOT <https://github.com/sarabellei/rtd_tutorial/edit/main/docs/source/index.rst>`_ providing essential functionality for the analysis of the vast amounts of very complex data produced by the experimental high-energy physics community, enabling interactive exploration, dynamic interoperability and rapid prototyping capabilities to C++ developers. It was first released in 2014 with the aim to facilitate the processing of scientific data in the field of high-energy physics as the interactive, C++ interpreter in  ROOT. 
ROOT is an open-source program written primarily in C++, developed by research groups in high-energy physics including `CERN <https://home.cern/>`_, `FERMILAB <https://www.fnal.gov/>`_  and `Princeton <https://www.princeton.edu/>`_ , and now used by most high-energy physics experiments. CERN is an European research organization that operates the largest particle physics laboratory in the world. Its experiments collect petabytes of data per year to be serialized, analyzed, and visualized as C++ objects.


Interactivity in C++ with Cling
-----------------------------------
**Interactive programming** is a programming approach allowing developers to change and modify the program as it runs. The final result is a program that actively responds to a developers’ intuitions, allowing them to make changes in their code, and to see the result of these changes without interrupting the running program. Interactive programming gives programmers the freedom to explore different scenarios while developing software, writing one expression at a time, figuring out what to do next at each step, and enabling them to quickly identify and fix bugs whenever they arise.
As an example, the High-Energy Physics community includes professionals with a variety of backgrounds, including physicists, nuclear engineers, and software engineers. **Cling** allows for interactive data analysis in `ROOT <https://github.com/sarabellei/rtd_tutorial/edit/main/docs/source/index.rst>`_  by giving researchers a way to prototype their C++ code, allowing them to tailor it to the particular scope of the analysis they want to pursue on a particular set of data before being added to the main framework.


**Interpreted language** is a way to achieve interactive programming. In statically compiled language, all source code is converted into native machine code and then executed by the processor before being run. An interpreted language instead runs through source programs line by line, taking an executable segment of source code, turning it into machine code, and then executing it. With this approach, when a change is made by the programmer, the interpreter will convey it without the need for the entire source code to be manually compiled. Interpreted languages are flexible, and offer features like dynamic typing and smaller program size. 

**Cling** allows C++, a language designed to be compiled, to be interpreted. When using **Cling**, the programmer benefits from both the power of C++ language, such as high-performance, robustness, fastness, efficiency, versatility, and the capability of an interpreter, which allows for interactive exploration and on-the-fly inspection of the source-code. 

Implementation Overview:
-----------------------------------
`LLVM <https://llvm.org/>`_ is a free, open-source compiler infrastructure under the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_. It is designed as a collection of tools including Front Ends parsers, Middle Ends optimizers, and Back Ends to produce machine code out of those programs. 

`Clang <https://clang.llvm.org/>`_  is a front-end that uses a **LLVM** license. **Clang** works by taking the source language (e.g. C++) and translating it into an intermediate representation that is then received by the compiler back end (i.e., the **LLVM** backend). Its library-based architecture makes it relatively easy to adapt **Clang** and build new tools based on it.  **Cling** inherits a number of features from **LLVM** and **Clang**, such as: fast compiling and low memory use, efficient C++ parsing, extremely clear and concise diagnostics, Just-In-Time compilation, pluggable optimizers, and support for `GCC <https://gcc.gnu.org/>`_  extensions. 

Interpreters allow for exploration of software development at the rate of human thought. Nevertheless, interpreter code can be slower than compiled code due to the fact that translating code at run time adds to the overhead and therefore causes the execution speed to be slower. This issue is overcome by exploiting the *Just-In-Time* (`JIT <https://en.wikipedia.org/wiki/Just-in-time_compilation>`_) compilation method. With the **JIT** approach, assembly is generated by the interpreter with a pseudocode (an `intermediate language <https://en.wikipedia.org/wiki/Common_Intermediate_Language>`_ that is not understood by the computer). The intermediate code is then translated into machine language when required for use. 
By following the **JIT** approach, **Cling** is able to evaluate whether a certain part of the source code is executed often, and then compile this part, therefore reducing the overall execution time.


Cling is (also, but not only) REPL:
-----------------------------------
A `read-eval-print-loop <https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop>`_ (**REPL**) is an interactive programming environment that takes user inputs, executes them, and returns the result to the user. In order to enable interactivity in C++, **Cling** provides several extensions to the C++ language:
Defining functions in the global scope: **Cling** redefines expressions at a global level. C++ provides limited support for this, **Cling** possesses the necessary semantics to re-define code while the program is running, minimizing the impedance mismatch between the **REPL** and the C++ codebase, and allowing for a seamlessly interactive programing experience.

Allows for implementation of commands which provide information about the current state of the environment. eg.., has an `Application Programming Interface <https://en.wikipedia.org/wiki/API>`_ (**API**) to provide information about the current state of the environment.

*Error recovery*: **Cling** has an efficient error recovery system which allows it to–  handle the errors made by the user without restarting or having to redo everything from the beginning.

*Tight feedback loop*: It provides feedback about the results of the developer’s choices that is both accurate and fast. 

*Facilitates debugging*: The programmer can inspect the printed result before deciding what expression to provide for the next line of code.

C++ in Jupyter Notebook - Xeus Cling:
-----------------------------------
The Jupyter Notebook technology allows users to create and share documents that contain live code, equations, visualizations and narrative text. It enables data scientists to easily exchange ideas or collaborate by sharing their analyses in a straight-forward and reproducible way.Jupyter’s official C++ kernel (Xeus-Cling) relies on Xeus, a C++ implementation of the kernel protocol, and Cling.
Using C++ in the Jupyter environment yields a different experience to C++ users. For example, Jupyter’s visualization system can be used to render rich content such as images, therefore bringing more interactivity into the Jupyter’s world.


Interactive CUDA C++ with Cling: 
-----------------------------------
The Cling CUDA extension brings the workflows of interactive C++ to GPUs, without losing performance and compatibility to existing software.
Through this extension, C++ and CUDA can be used interactively on the target machine, allowing for optimization for particular models of accelerator hardware. The extension can be run on a Jupyter setup. Cling CUDA found application in the field of modeling of high-energy particles and radiation produced by high-energy laser facilities. In this framework, Cling CUDA allows for an interactive approach which enables relaunching only a wanted part of a simulation, starting from a given point which can be decided by the user.


Conclusion:
-----------------------------------
**Cling** is not just a **REPL**, it is an embeddable and extensible execution system for efficient incremental execution of C++. **Cling** allows us to decide how much we want to compile statically and how much to defer for the target platform. **Cling** enables reflection and introspection information in high-performance systems such as **ROOT**, or **Xeus Jupyter**, where it provides efficient code for performance-critical tasks where hot-spot regions can be annotated with specific optimization levels. We will see more concrete examples in the slides to follow. 


You can find a detailed explanation of Cling’s design in the following paper: V Vasilev et al 2012 J. Phys.: Conf. Ser. 396 052071
More in detail, the paper describes in detail Cling’s characteristic features  such as syntactic and semantic error recovery, execution of statements, loading of dynamic objects (i.e. external objects loaded at runtime), entity redefinition, and displaying of execution results.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.


Table of Contents
--------

 .. toctree::
    :numbered:
    
    background
    interactivity
    implementation
    REPL
    XEUS
    cudaC++
    references

