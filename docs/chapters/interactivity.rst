Interactivity in C++ with Cling
-------------------------------

**Interactive programming** is a programming approach that allows developers to
 change and modify the program as it runs. The final result is a program that
 actively responds to a developers’ intuitions, allowing them to make changes in
 their code, and to see the result of these changes without interrupting the
 running program. Interactive programming gives programmers the freedom to
 explore different scenarios while developing software, writing one expression
 at a time, figuring out what to do next at each step, and enabling them to
 quickly identify and fix bugs whenever they arise.  As an example, the
 High-Energy Physics community includes professionals with a variety of
 backgrounds, including physicists, nuclear engineers, and software
 engineers. Cling allows for interactive data analysis in `ROOT
 <https://root.cern/>`_ by giving researchers a way to prototype their C++ code,
 allowing them to tailor it to the particular scope of the analysis they want to
 pursue on a particular set of data before being added to the main framework.

**Interpreted language** is a way to achieve interactive programming. In
 statically compiled language, all source code is converted into native machine
 code and then executed by the processor before being run. An interpreted
 language instead runs through source programs line by line, taking an
 executable segment of source code, turning it into machine code, and then
 executing it. With this approach, when a change is made by the programmer, the
 interpreter will convey it without the need for the entire source code to be
 manually compiled. Interpreted languages are flexible, and offer features like
 dynamic typing and smaller program size.

**Cling** is not an interpreter, it is a Just-In-Time (JIT) compiler that feels
 like an interpreter, and allows C++, a language designed to be compiled, to be
 interpreted. When using Cling, the programmer benefits from both the power of
 C++ language, such as high-performance, robustness, fastness, efficiency,
 versatility, and the capability of an interpreter, which allows for interactive
 exploration and on-the-fly inspection of the source-code.
