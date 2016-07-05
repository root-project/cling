Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.2. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the
main Cling web page, this document applies to the *next* release, not
the current one.

What's New in Cling 0.2?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

Major New Features
------------------

- Switch Cling's execution engine to use LLVM's Orc JIT. The new JIT allows
Cling to support:
  - ARM and PowerPC architectures;
  - Exceptions. Cling can throw and catch exceptions from interpreted and
  compiled code.
    ```cpp
    [cling]$ throw new std::exception();
    Exception occurred. Recovering...
    ```
  - Inline assembly. Cling can execute `asm` statements.
    ```cpp
    [cling]$ extern "C" int printf(const char*, ...);
    [cling]$ int arg1=1, arg2=2, add;
    [cling]$ asm ("addl %%ebx, %%eax;" : "=a" (add) : "a" (arg1) , "b" (arg2));
    [cling]$ printf( "%d + %d = %d\n", arg1, arg2, add );
    1 + 2 = 3
    ```
- Reduce memory usage caused by excessive memory allocations in Cling lookup
routines.
- Stabilize error recovery caused by handling of templated declarations.
- Implement a user-extendable value streaming engine.
- Implement shebang support.
- Protect against invalid pointer dereferences.
  ```cpp
  [cling]$ struct S{ int a; };
  [cling]$ ((S*)0x123)->a;
  input_line_4:2:2: warning: invalid memory pointer passed to a callee:
  ((S*)0x123)->a;
  ^~~~~~~~~~~
  [cling]$ ((S*)0)->a;
  input_line_5:2:2: warning: null passed to a callee that requires a non-null argument [-Wnonnull]
  ((S*)0)->a;
  ^~~~~~~
  ```
- Redirect stderr and stdout. Cling's users can redirect the output streams.
  ```cpp
  [cling]$ // Redirects stdout to /tmp/outfile.txt
  [cling]$ .> /tmp/outfile.txt
  [cling]$ // Toggles back to the prompt
  [cling]$ .>
  [cling]$ // Redirects stderr to /tmp/errfile.txt
  [cling]$ .2> /tmp/errfile.txt
  [cling]$ // Redirects stdout and stderr to /tmp/bothfile.txt
  [cling]$ .&> /tmp/bothfile.txt
  ```
- Add `#pragma cling` directives:
  - `#pragma cling add_include_path("/include/path/")`
  - `#pragma cling add_library_path("/library/path/")`
  - `#pragma cling load("libUserLib")`
- Add `.@` metacommand to cancel multiline user input.
- Add `.debug [Constant]` metacommand to tune the generated debug information.

Cling as a Library
------------------
- Implement a generator of forward declarations from a given AST declaration.
- Improve string-to-decl lookup helper.
- Support of parent-children (multiple) interpreters.

External Dependencies
---------------------
- Upgrade to LLVM r272382.
- Upgrade to CMake 3.4.3 (following LLVM requirements)

Misc
------
- Add a package building script [Cling Packaging Tool (CPT)](../tools/packaging)
which can build Cling from source and generate installer bundles for a wide
range of platforms.
- Add [Jupyter kernel](../tools/Jupyter/kernel).

Experimental Features
---------------------
- Code unloading:
  - Generate one module per transaction, which simplifies unloading of
  llvm::Modules and machine code.
  - Add support for unloading few STL header files.
- Dynamic Scopes:
  - Handle of errors occurred when synthesizing AST nodes.

Fixed Bugs:
----------
[comment]: <> ( Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' )
[comment]: <> ( Standard MarkDown doesn't support neither variables nor <base> )

[ROOT-4760](https://sft.its.cern.ch/jira/browse/ROOT-4760)
[ROOT-5467](https://sft.its.cern.ch/jira/browse/ROOT-5467)
[ROOT-5607](https://sft.its.cern.ch/jira/browse/ROOT-5607)
[ROOT-5698](https://sft.its.cern.ch/jira/browse/ROOT-5698)
[ROOT-5844](https://sft.its.cern.ch/jira/browse/ROOT-5844)
[ROOT-6153](https://sft.its.cern.ch/jira/browse/ROOT-6153)
[ROOT-6185](https://sft.its.cern.ch/jira/browse/ROOT-6185)
[ROOT-6345](https://sft.its.cern.ch/jira/browse/ROOT-6345)
[ROOT-6365](https://sft.its.cern.ch/jira/browse/ROOT-6365)
[ROOT-6385](https://sft.its.cern.ch/jira/browse/ROOT-6385)
[ROOT-6611](https://sft.its.cern.ch/jira/browse/ROOT-6611)
[ROOT-6625](https://sft.its.cern.ch/jira/browse/ROOT-6625)
[ROOT-6650](https://sft.its.cern.ch/jira/browse/ROOT-6650)
[ROOT-6692](https://sft.its.cern.ch/jira/browse/ROOT-6692)
[ROOT-6695](https://sft.its.cern.ch/jira/browse/ROOT-6695)
[ROOT-6705](https://sft.its.cern.ch/jira/browse/ROOT-6705)
[ROOT-6712](https://sft.its.cern.ch/jira/browse/ROOT-6712)
[ROOT-6719](https://sft.its.cern.ch/jira/browse/ROOT-6719)
[ROOT-6755](https://sft.its.cern.ch/jira/browse/ROOT-6755)
[ROOT-6791](https://sft.its.cern.ch/jira/browse/ROOT-6791)
[ROOT-6824](https://sft.its.cern.ch/jira/browse/ROOT-6824)
[ROOT-6832](https://sft.its.cern.ch/jira/browse/ROOT-6832)
[ROOT-6909](https://sft.its.cern.ch/jira/browse/ROOT-6909)
[ROOT-6942](https://sft.its.cern.ch/jira/browse/ROOT-6942)
[ROOT-6976](https://sft.its.cern.ch/jira/browse/ROOT-6976)
[ROOT-7031](https://sft.its.cern.ch/jira/browse/ROOT-7031)
[ROOT-7037](https://sft.its.cern.ch/jira/browse/ROOT-7037)
[ROOT-7041](https://sft.its.cern.ch/jira/browse/ROOT-7041)
[ROOT-7090](https://sft.its.cern.ch/jira/browse/ROOT-7090)
[ROOT-7092](https://sft.its.cern.ch/jira/browse/ROOT-7092)
[ROOT-7095](https://sft.its.cern.ch/jira/browse/ROOT-7095)
[ROOT-7114](https://sft.its.cern.ch/jira/browse/ROOT-7114)
[ROOT-7159](https://sft.its.cern.ch/jira/browse/ROOT-7159)
[ROOT-7163](https://sft.its.cern.ch/jira/browse/ROOT-7163)
[ROOT-7184](https://sft.its.cern.ch/jira/browse/ROOT-7184)
[ROOT-7269](https://sft.its.cern.ch/jira/browse/ROOT-7269)
[ROOT-7276](https://sft.its.cern.ch/jira/browse/ROOT-7276)
[ROOT-7295](https://sft.its.cern.ch/jira/browse/ROOT-7295)
[ROOT-7310](https://sft.its.cern.ch/jira/browse/ROOT-7310)
[ROOT-7364](https://sft.its.cern.ch/jira/browse/ROOT-7364)
[ROOT-7426](https://sft.its.cern.ch/jira/browse/ROOT-7426)
[ROOT-7462](https://sft.its.cern.ch/jira/browse/ROOT-7462)
[ROOT-7614](https://sft.its.cern.ch/jira/browse/ROOT-7614)
[ROOT-7619](https://sft.its.cern.ch/jira/browse/ROOT-7619)
[ROOT-7673](https://sft.its.cern.ch/jira/browse/ROOT-7673)
[ROOT-7737](https://sft.its.cern.ch/jira/browse/ROOT-7737)
[ROOT-7744](https://sft.its.cern.ch/jira/browse/ROOT-7744)
[ROOT-7837](https://sft.its.cern.ch/jira/browse/ROOT-7837)
[ROOT-7840](https://sft.its.cern.ch/jira/browse/ROOT-7840)
[ROOT-7918](https://sft.its.cern.ch/jira/browse/ROOT-7918)
[ROOT-8034](https://sft.its.cern.ch/jira/browse/ROOT-8034)
[ROOT-8056](https://sft.its.cern.ch/jira/browse/ROOT-8056)
[ROOT-8096](https://sft.its.cern.ch/jira/browse/ROOT-8096)
[ROOT-8111](https://sft.its.cern.ch/jira/browse/ROOT-8111)


[comment]: <> (Additional Information)
[comment]: <> (----------------------)
[comment]: <> (A wide variety of additional information is available on the
[Cling web page]\(http://root.cern/cling\). The web page contains versions of
the API documentation which are up-to-date with the git version of the source
code. You can access versions of these documents specific to this release by
going into the “clang/docs/” directory in the Cling source tree.)
[comment]: <> (If you have any questions or comments about Cling, please feel
free to contact us via the mailing list.)
