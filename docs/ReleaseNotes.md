Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 1.3. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main Cling
web page, this document applies to the *next* release, not the current one.

What's New in Cling 1.3?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM20.
* Module-map and standard-header housekeeping to track modern C++ standards:
  several headers deprecated in C++17 or not part of C++20 were removed from
  the shipped module maps (for example: ciso646, ccomplex, cstdalign,
  cstdbool, ctgmath) to avoid spurious build warnings when using newer
  compilers.
* Darwin/macOS SDK compatibility: darwin modulemaps were adapted to macOS SDK
  15.4 (with separate handling for macOS 15.2/15.4 overlays).
* Clad and LLVM integration improvements (Clad bumped; fixes for building with
  builtin_llvm=OFF and options to provide CLAD_SOURCE_DIR).


Major New Features
------------------

* Improved, clang-repl-style code completion -- Cling now integrates an
  autocomplete approach that mirrors clang-repl’s design, avoiding the need
  for a nested interpreter and making completion faster and lighter-weight.
  This includes CI support for an autocomplete mode and unit tests for the
  functionality.
* Safer dictionary / JIT symbol handling on macOS -- Cling injects
  missing compiler-rt complex-division symbols into the JIT symbol table on
  macOS and now recognizes Mach-O bundle (MH_BUNDLE) files as valid shared
  libraries so module-built dictionary libraries load correctly.
* Build & usage documentation and standalone build improvements -- clearer
  instructions for building Cling both as a standalone project and integrated
  with LLVM; README and README links updated to reduce user confusion.
* Backport / upstream sync for template & AST improvements -- lazy template
  loading behavior has been updated to align with upstream Clang/LLVM changes
  (backport of upstream lazy-template-loading support).


Misc
----

* CMake and build-system updates -- minimum CMake requirement raised to 3.10
to avoid deprecation warnings and better match modern build toolchains.
* Clad version bump / packaging fixes -- bundled Clad updated (bumped to
  v1.9) and the Clad build path handling improved so -DCLAD_SOURCE_DIR works
  correctly and avoids unnecessary network git checkouts during cmake.
* Platform support and targets -- initial support for Loong64 added to the
  build system.


Fixed Bugs
----------

[17515](https://github.com/root-project/root/issues/17515)
[18236](https://github.com/root-project/root/issues/18236)
[19404](https://github.com/root-project/root/issues/19404)
[19450](https://github.com/root-project/root/issues/19450)
[20063](https://github.com/root-project/root/issues/20063)
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

Devajith Valaparambil Sreeramaswamy (44)
ferdymercury (15)
Jonas Rembser (10)
Jonas Hahnfeld (9)
Vassil Vassilev (7)
Danilo Piparo (5)
Bertrand Bellenot (4)
Philippe Canal (3)
Yeoh Joer (2)
Zhou Qiankang (1)
Vipul Cariappa (1)
Surya Somayyajula (1)
Sandro Wenzel (1)
saisoma123 (1)
Mattias Ellert (1)
Lukas Breitwieser (1)
jeffbla (1)
Erik Jensen (1)
edish-github (1)
Chris Burr (1)
chn (1)
Axel Naumann (1)
Aaron Jomy (1)
