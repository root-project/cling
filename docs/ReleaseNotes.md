Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 1.2. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main Cling
web page, this document applies to the *next* release, not the current one.

What's New in Cling 1.2?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM18.

Major New Features
------------------
* Improvements in stability

Misc
----
* Better handling of llvm::Error
* Better integration with Clad
* Modulemap fixes

Experimental Features
---------------------
* An experimental feature

Jupyter
-------
* A Jupyter feature


Fixed Bugs
----------
[16654](https://github.com/root-project/cling/issues/16654)

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

Devajith Valaparambil Sreeramaswamy (22)
Jonas Hahnfeld (18)
Jonas Rembser (2)
Bertrand Bellenot (2)
ferdymercury (1)
dbonner (1)
Vipul Cariappa (1)
Vassil Vassilev (1)
Fredrik (1)
Danilo Piparo (1)
Aaron Jomy (1)
