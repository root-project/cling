Why interpreting C++ with Cling?
-----------------------------------

1. **Learning C++:**
   
One use case of Cling is to aid the C++ learning process. Offering imediate
feedback the user can easily get familiar with the structures and spelling of
the language.


2. **Creating scripts:**
   
The power of an interpreter lays as well in the compactness and ease of
repeatedly running a small snippet of code - aka a script. This can be done in
Cling by inserting the bash-like style line:

.. code:: bash
   
   #!/usr/bin/cling
   
3. **Rapid Application Development (RAD):**

Cling can be used successfully for Rapid Application Development allowing for
prototyping and proofs of concept taking advantage of dynamicity and feedback
during the implementation process.

4. **Runtime-Generated Code**

Sometime it's convenient to create code as a reaction to input
(user/network/configuration). Runtime-generated code can interface with C++
libraries.

5. **Embedding Cling:**

The functionality of an application can be enriched by embedding Cling. To embed
Cling, the main program has to be provided. One of the things this main program
has to do is initialize the Cling interpreter. There are optional calls to pass
command line arguments to Cling. Afterwards, you can call the interpreter from
any anywhere within the application.

For compilation and linkage the application needs the path to the Clang and LLVM
libraries and the invocation is order dependent since the linker cannot do
backward searches.


.. code:: bash

   g++ embedcling.cxx -std=c++11 -L/usr/local/lib
                    -lclingInterpreter -lclingUtils 
                    -lclangFrontend -lclangSerialization -lclangParse -lclangSema 
                    -lclangAnalysis -lclangEdit -lclangLex -lclangDriver -lclangCodeGen 
                    -lclangBasic  -lclangAST  
                    `llvm-config 
                      --libs bitwriter mcjit orcjit native option 
                        ipo profiledata instrumentation objcarcopts` 
                      -lz -pthread -ldl -ltinfo 
                    -o embedcling
                    

Embedding Cling requires the creation of the interpreter. Optionally compiler
arguments and the resource directory of LLVM can be passed. An example is the
following:


.. code:: bash

   #include "cling/Interpreter/Interpreter.h"

   int main(int argc, char** argv) {
      const char* LLVMRESDIR = "/usr/local/"; //path to llvm resource directory
      cling::Interpreter interp(argc, argv, LLVMRESDIR);

      interp.declare("int p=0;");
    }
        
A more complete example could be found in `<tools/demo/cling-demo.cpp>`_.
