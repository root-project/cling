Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.8. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-project/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 0.8?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM-5.

Major New Features
------------------

Misc
----
* Improve in the C++ modules support
* Fix issues in the definition shadowing
* Improve the integration with clad


Fixed Bugs
----------

[ROOT-10886](https://sft.its.cern.ch/jira/browse/ROOT-10886)
[ROOT-7199](https://sft.its.cern.ch/jira/browse/ROOT-7199)

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

Vassil Vassilev (7)
Pratyush Das (4)
Javier Lopez-Gomez (4)
Axel Naumann (4)
Philippe Canal (2)
Vaibhav Garg (1)
Sylvain Corlay (1)
Henry Schreiner (1)
