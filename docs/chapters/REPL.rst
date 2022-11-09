Cling is (also, but not only) REPL
-----------------------------------

A `read-eval-print-loop <https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop>`_
(**REPL**) is an interactive programming environment that takes user inputs,
executes them, and returns the result to the user. In order to enable
interactivity in C++, Cling provides several extensions to the C++ language:

1. **Defining functions in the global scope:** Cling redefines expressions at a
   global level. C++ provides limited support for this, Cling possesses the
   necessary semantics to re-define code while the program is running,
   minimizing the impedance mismatch between the **REPL** and the C++ codebase,
   and allowing for a seamlessly interactive programing experience.

2. **Allows for implementation of commands** that provide information about the
   current state of the environment. e.g., has an `Application Programming
   Interface <https://en.wikipedia.org/wiki/API>`_ (**API**) to provide
   information about the current state of the environment.

3. **Error recovery:** Cling has an efficient error recovery system which allows
   it to handle the errors made by the user without restarting or having to redo
   everything from the beginning.

4. **Tight feedback loop:** It provides feedback about the results of the
   developer’s choices that is both accurate and fast.

5. **Facilitates debugging:** The programmer can inspect the printed result
   before deciding what expression to provide for the next line of code.
