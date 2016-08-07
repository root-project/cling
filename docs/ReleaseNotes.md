Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.3. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the
main Cling web page, this document applies to the *next* release, not
the current one.

What's New in Cling 0.3?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

Major New Features
------------------
* Support for GCC5 ABI, enabling e.g. native support for newer Fedora and Ubuntu.
* Code completion from the prompt.
* Rudimentary Windows support, requiring CMake nightly builds later than
  01.08.2016 and MSVC 2015.
* Enable C++ module builds.

Cling as a Library
------------------
* Support for building cling as a shared library.
* Improved CMake dependencies.

External Dependencies
---------------------
* Upgrade to LLVM r274612.

Misc
----
* Extend Cling's static web site.

Experimental Features
---------------------
* Code unloading:
* Dynamic Scopes:

Fixed Bugs
----------
<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Standard MarkDown doesn't support neither variables nor <base> --->


<!---Additional Information
----------------------
A wide variety of additional information is available on the
[Cling web page](http://root.cern/cling). The web page contains versions of
the API documentation which are up-to-date with the git version of the source
code. You can access versions of these documents specific to this release by
going into the “clang/docs/” directory in the Cling source tree.

If you have any questions or comments about Cling, please feel free to contact
us via the mailing list.--->
