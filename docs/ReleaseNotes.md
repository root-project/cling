Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.7. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-project/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 0.7?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

Major New Features
------------------
* Implement a mechanism allowing to redefine entities with the same name -- the
  *DefinitionShadower* is not default and can be turned on by:
  ```cpp
  [cling] #include "cling/Interpreter/Interpreter.h"
  [cling] gCling->allowRedefinition()
  ```
* Improve CUDA support:
  - Replace the PTX compiler with the internal one.
  - Add in-memory fatbin generation.
* Initial Apple Silicon Support
* Tighter Clang C++ Modules integration:
  - Implement global module indexing to improve module loading.
  - Automatic virtual overlay files for libc, std, tinyxml, boost and cuda.
* Implement dynamic library symbol resolver based on the binary object formats.

Misc
----
* Improvements in the cpt packaging system and travis continuous integration.
* Windows improvements of jitted variables 


Fixed Bugs
----------
[Cling-197](https://github.com/root-project/cling/issues/197)
[Cling-284](https://github.com/root-project/cling/issues/284)
[Cling-297](https://github.com/root-project/cling/issues/297)

[ROOT-10193](https://sft.its.cern.ch/jira/browse/ROOT-10193)
[ROOT-10224](https://sft.its.cern.ch/jira/browse/ROOT-10224)
[ROOT-10285](https://sft.its.cern.ch/jira/browse/ROOT-10285)
[ROOT-10333](https://sft.its.cern.ch/jira/browse/ROOT-10333)
[ROOT-10354](https://sft.its.cern.ch/jira/browse/ROOT-10354)
[ROOT-10426](https://sft.its.cern.ch/jira/browse/ROOT-10426)
[ROOT-10499](https://sft.its.cern.ch/jira/browse/ROOT-10499)
[ROOT-10511](https://sft.its.cern.ch/jira/browse/ROOT-10511)
[ROOT-10677](https://sft.its.cern.ch/jira/browse/ROOT-10677)
[ROOT-10689](https://sft.its.cern.ch/jira/browse/ROOT-10689)
[ROOT-10751](https://sft.its.cern.ch/jira/browse/ROOT-10751)
[ROOT-10777](https://sft.its.cern.ch/jira/browse/ROOT-10777)
[ROOT-10791](https://sft.its.cern.ch/jira/browse/ROOT-10791)
[ROOT-10798](https://sft.its.cern.ch/jira/browse/ROOT-10798)
[ROOT-10803](https://sft.its.cern.ch/jira/browse/ROOT-10803)
[ROOT-10812](https://sft.its.cern.ch/jira/browse/ROOT-10812)
[ROOT-10917](https://sft.its.cern.ch/jira/browse/ROOT-10917)


<!---Additional Information
----------------------
A wide variety of additional information is available on the
[Cling web page](http://root.cern/cling). The web page contains versions of
the API documentation which are up-to-date with the git version of the source
code. You can access versions of these documents specific to this release by
going into the “clang/docs/” directory in the Cling source tree.

If you have any questions or comments about Cling, please feel free to contact
us via the mailing list.--->


Special Kudos
=============
This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

Vassil Vassilev (57)
Pratyush Das (35)
Axel Naumann (28)
Simeon Ehrig (13)
Javier Lopez-Gomez (13)
Vaibhav Garg (8)
Philippe Canal (7)
Bertrand Bellenot (7)
Frederich Munch (3)
Chris Burr (2)
pankaj kumar (1)
Stephan Hageboeck (1)
Sergey Linev (1)
Oksana Shadura (1)
Martin Ritter (1)
Jonas Hahnfeld (1)
Enrico Guiraud (1)
Alexander Penev (1)
