Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.9. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-project/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 0.9?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM 9. LLVM 9 brings to cling better stability, full C++17 support
  and CUDA 10 support.


Misc
----
* Improve CUDA device compiler such as enabling sm level configuration and
  and add `--cuda-path` option.
* Improve the DefinitionShadower with respect to C++ Modules
* Embed [Vc](https://github.com/VcDevel/Vc) modulemap file
* Build the available cling plugins by default.
* Reduce dependence on custom clang patches.


Fixed Bugs
----------
[ROOT-10962](https://sft.its.cern.ch/jira/browse/ROOT-10962)
[ROOT-7775](https://sft.its.cern.ch/jira/browse/ROOT-7775)
[ROOT-10703](https://sft.its.cern.ch/jira/browse/ROOT-10703)
[ROOT-10962](https://sft.its.cern.ch/jira/browse/ROOT-10962)
[ROOT-GH-7021](https://github.com/root-project/root/issues/7021)
[ROOT-GH-7090](https://github.com/root-project/root/issues/7090)
[ROOT-GH-7657](https://github.com/root-project/root/issues/7657)
[CLING-GH-399](https://github.com/root-project/cling/issues/399)


Special Kudos
=============
This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

Vassil Vassilev (80)
Axel Naumann (29)
Simeon Ehrig (9)
Pratyush Das (7)
Jonas Hahnfeld (7)
Javier Lopez-Gomez (1)
Enrico Guiraud (1)
David (1)
Bertrand Bellenot (1)
