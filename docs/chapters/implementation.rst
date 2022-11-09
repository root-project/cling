Used Technology
---------------

`LLVM <https://llvm.org/>`_ is a free, open-source compiler infrastructure under
the `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_. It is
designed as a collection of tools including Front Ends parsers, Middle Ends
optimizers, and Back Ends to produce machine code out of those programs.

`Clang <https://clang.llvm.org/>`_ is a front-end that uses a LLVM
license. Clang works by taking the source language (e.g. C++) and translating it
into an intermediate representation that is then received by the compiler back
end (i.e., the LLVM backend). Its library-based architecture makes it relatively
easy to adapt Clang and build new tools based on it.  Cling inherits a number of
features from LLVM and Clang, such as: fast compiling and low memory use,
efficient C++ parsing, extremely clear and concise diagnostics, Just-In-Time
compilation, pluggable optimizers, and support for `GCC <https://gcc.gnu.org/>`_
extensions.


Interpreters allow for exploration of software development at the rate of human
thought. Nevertheless, interpreter code can be slower than compiled code due to
the fact that translating code at run time adds to the overhead and therefore
causes the execution speed to be slower. This issue is overcome by exploiting
the *Just-In-Time* (`JIT
<https://en.wikipedia.org/wiki/Just-in-time_compilation>`_) compilation method,
which allows an efficient meory management (for example, by evaluating whether a
certain part of the source code is executed often, and then compile this part,
therefore reducing the overall execution time).

With the JIT approach, the developer types the code in Cling's command
prompt. The input code is then lowered to Clang, where is compiled and
eventually transformed in order to attach specific behavior. Clang compiles then
the input into an AST representation, that is then lowered to LLVM IR, an
`intermediate language
<https://en.wikipedia.org/wiki/Common_Intermediate_Language>`_ that is not
understood by the computer. LLVM’s just-in-time compilation infrastructure
translates then the intermediate code into machine language (eg. Intel x86 or
NVPTX) when required for use.  Cling's JIT compiler relies on LLVM's project
`ORC <https://llvm.org/docs/ORCv2.html>`_ (On Request Compilation) Application
Programming Interfaces (APIs).
