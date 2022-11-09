Literature
=====


.. list-table:: What is Cling?
   :widths: 25 25 50
   :header-rows: 1

   * - Link
     - Info 
     - Description
   * - `Relaxing the One Definition Rule in Interpreted C++ <https://dl.acm.org/doi/10.1145/3377555.3377901>`_
     - *Javier Lopez Gomez et al.*
       
       29th International Conference on Compiler Construction 2020
     - This paper discusses how Cling enables redefinitions of C++ entities at the prompt, and the implications of interpreting C++ and the One Definition Rule (ODR) in C++
   * - `Cling – The New Interactive Interpreter for ROOT 6 <https://iopscience.iop.org/article/10.1088/1742-6596/396/5/052071>`_
     - *V Vasilev et al* 2012 J. Phys.: Conf. Ser. 396 052071
     - This paper describes the link between Cling and ROOT. The concepts of REPL and  JIT compilation. Cling’s methods for handling errors, expression evaluation, streaming out of execution results, runtime dynamism.
   * - `Interactive, Introspected C++ at CERN <https://www.youtube.com/watch?v=K2KqEV866Ro>`_
     - *V Vasilev*, CERN PH-SFT, 2013
     - Vassil Vasilev (Princeton University) explains how Cling enables interactivity in C++, and  illustrates the type introspection mechanism provided by the interpreter.
   * - `Introducing Cling, a C++ Interpreter Based on Clang/LLVM <https://www.youtube.com/watch?v=f9Xfh8pv3Fs>`_
     - *Axel Naumann* 2012  Googletechtalks
     - Axel Naumann (CERN) discusses Cling’s most relevant features: abstract syntax tree (AST) production, wrapped functions, global initialization of a function, delay expression evaluation at runtime, and dynamic scopes.
   * - `Creating Cling, an interactive interpreter interface <https://www.youtube.com/watch?v=BjmGOMJWeAo>`_
     - *Axel Naumann* 2010 LLVM Developers’ meeting
     - This presentation introduces Cling, an ahead-of-time compiler that extends C++ for ease of use as an interpreter.
  

   
.. list-table:: Demos, tutorials, Cling’s ecosystem:
   :widths: 25 25 50
   :header-rows: 1

   * - Link
     - Info 
     - Description
   * - `Cling integration | CLion <https://www.jetbrains.com/help/clion/cling-integration.html#install-cling>`_
     - 2022.2 Version
     - CLion uses Cling to integrate the  `Quick Documentation <https://www.jetbrains.com/help/clion/2022.2/viewing-inline-documentation.html>`_ popup by allowing you to view the value of the expressions evaluated at compile time.
   * - `Interactive C++ for Data Science <https://www.youtube.com/watch?v=23E0S3miWB0&t=2716s>`_
     - *Vassil Vassilev* 2021 CppCon (The C++ Conference)
     - In this video, the author discusses how Cling enables interactive C++ for Data Science projects. 
   * - `Cling -- Beyond Just Interpreting C++ <https://blog.llvm.org/posts/2021-03-25-cling-beyond-just-interpreting-cpp/>`_
     - *Vassil Vassilev* 2021 The LLVM Project Blog
     - This blog page discusses how Cling enables template Instantiation on demand, language interoperability on demand, interpreter/compiler as a service, plugins extension.
   * - `TinySpec-Cling <https://github.com/nwoeanhinnogaehr/tinyspec-cling>`_
     - Noah Weninger 2020
     - A tiny C++ live-coded overlap-add (re)synthesizer for Linux, which uses Cling to add REPL-like functionality for C++ code.
   * - `Interactive C++ for Data Science <https://blog.llvm.org/posts/2020-12-21-interactive-cpp-for-data-science/>`_
     - *Vassil Vassilev,* *David Lange,* *Simeon Ehrig,* *Sylvain Corlay* 2020 The LLVM Project Blog
     - Cling enables eval-style programming for Data Science applications. Examples of ROOT and Xeus-Cling for data science are shown.
   * - `Interactive C++ with Cling <https://blog.llvm.org/posts/2020-11-30-interactive-cpp-with-cling/>`_
     - *Vassil Vassilev* 2020 The LLVM Project Blog
     - This blog page briefly discusses the concept of interactive C++ by presenting Cling’s main features, such as wrapper functions, entity redefinition, error recovery. 
   * - `Using the Cling C++ Interpreter on the Bela Platform <https://gist.github.com/jarmitage/6e411ae8746c04d6ecbee1cbc1ebdcd4>`_
     - Jack Armitage 2019
     - Cling has been installed on a BeagleBoard to bring live coding to the Bela interactive audio platform.
   * - `Implementation of GlobalModuleIndex in ROOT and Cling <https://indico.cern.ch/event/840376/contributions/3525646/attachments/1895398/3127159/GSoC_Presentation__GMI.pdf>`_
     - *Arpitha Raghunandan* 2012 Google Summer of Code GSoC
     - GlobalModuleIndex can be used for improving ROOT’s and Cling’s performance 
   * - `Example project using cling as library <https://github.com/root-project/cling/tree/master/tools/demo>`_
     - *Axel Naumann* 2016 GitHub
     - This video showcases how to use Cling as a library, and shows how to set up a simple CMake configuration that uses Cling.
   * - `Cling C++ interpreter testdrive <https://www.youtube.com/watch?v=1IGTHusaJ18>`_
     - *Mika* 2015 Youtube
     - In this tutorial, a developer tries Cling for the first time by uploading a few simple C++ user-cases onto Cling, involving also the loading of external files
   * - `Building an Order Book in C++ <https://www.youtube.com/watch?v=fxN4xEZvrxI>`_
     - *Dimitri Nesteruk* 2015 Youtube
     - This demo shows how to build a simple order book using C++, CLion, Google Test and, of course, Cling. 
   * - `Cling C++ interpreter testdrive <https://www.youtube.com/watch?v=1IGTHusaJ18>`_
     - Dimitri Nesteruk 2015 Youtube
     - This tutorial describes Cling’s general features. You will learn how to start Cling on Ubuntu, how to write a simple expression (N=5, N++) and how to define a Class for calculating body mass index. 
   * - `Cling Interactive OpenGL Demo <https://www.youtube.com/watch?v=eoIuqLNvzFs>`_
     - *Alexander Penev* 2012 Youtube
     - This demo shows how to use Cling for interactive OpenGL. A rotating triangle with changing color, a static figure, and a figure with light effects are created.
     
     

.. list-table:: Language Interoperability with Cling:
   :widths: 25 25 50
   :header-rows: 1

   * - Link
     - Info 
     - Description
   * - `Compiler Research - Calling C++ libraries from a D-written DSL: A cling/cppyy-based approach <https://www.youtube.com/watch?v=7teqrCNzrD8>`_
     - *Alexandru Militaru* 2021 Compiler-Research Meeting
     - This video presents D and C++ interoperability through SIL-Cling architecture



.. list-table:: Interactive CUDA C++ with Cling:
   :widths: 25 25 50
   :header-rows: 1

   * - Link
     - Info 
     - Description
   * - `Adding CUDA® Support to Cling: JIT Compile to GPUs <https://www.youtube.com/watch?v=XjjZRhiFDVs>`_
     - *Simeon Ehrig* 2020 LLVM Developer Meeting
     - Interactive CUDA-C++ through Cling is presented. Cling-CUDA architecture is discussed in detail, and an example of interactive simulation for laser plasma applications is shown. 



.. list-table:: C++ in Jupyter Notebook - Xeus Cling:
   :widths: 25 25 50
   :header-rows: 1
  
   * - Link
     - Info 
     - Description
   * - `Interactive C++ code development using C++Explorer and GitHub Classroom for educational purposes <https://www.youtube.com/watch?v=HBgF2Yr0foA>`_
     - *Patrick Diehl* 2020 Youtube
     - C++Explorer is a novel teaching environment based on Jupyterhub and Cling, adapted to teaching C++ programming and source code management.
   * - `Deep dive into the Xeus-based Cling kernel for Jupyter <https://www.youtube.com/watch?v=kx3wvKk4Qss>`_
     - *Vassil Vassilev* 2021 Youtube
     - Xeus-Cling is a Cling-based notebook kernel which delivers interactive C++. 
   * - `Xeus-Cling: Run C++ code in Jupyter Notebook <https://www.youtube.com/watch?v=4fcKlJ_5QQk>`_ 
     - *LearnOpenCV* 2019 Youtube
     - In this demo, you will learn an example of C++ code in Jupyter Notebook using Xeus-Cling kernel. 



.. list-table:: Clad:
   :widths: 25 25 50
   :header-rows: 1
  
   * - Link
     - Info 
     - Description
   * - `Clad: Automatic differentiation plugin for C++ <https://clad.readthedocs.io/en/latest/index.html>`_  
     - Read The Docs webpage
     - Clad is a plugin for Cling. It allows to perform Automatic Differentiation (AD) on multivariate functions and functor objects

