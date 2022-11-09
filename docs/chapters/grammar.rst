Command Line
============

Cling has its own command line, which looks like any other Unix shell. The
emacs-like command line editor is what we call interactive command line or
interactive shell.

Once we start Cling it automatically includes several header files and its own
runtime universe. Thus it creates the minimal environment for the user to start.


Grammar
-------

Cling is capable to parse everything that `Clang <https://clang.llvm.org/>`_ can
do. In addition, Cling can parse some interpreter-specific C++ extensions.

Metaprocessor
-------------

Cling Metaprocessor provides convenient and easy to use interface for changing
the interpreter’s internal state or for executing handy commands. Cling provides
the following metaprocessor commands:

**syntax: .(command)**, where command is:


.. code:: bash

    x filename.cxx
    
loads filename and calls void filename() if defined;


.. code:: bash

    L library | filename.cxx
    
loads library or filename.cxx;


.. code:: bash

   printAST
    
(DEBUG ONLY) shows the abstract syntax tree after each processed entity;


.. code:: bash

   I path
    
adds an include path;


.. code:: bash

   .@ 

Cancels the multiline input;


.. code:: bash

   .dynamicExtensions

Turns on cling's dynamic extensions. This in turn enables the dynamic lookup and
the late resolving of the identifier. With that option cling tries to heal the
compile-time failed lookups at runtime.
