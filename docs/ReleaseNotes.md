Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.4. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-mirror/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 0.4?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

Major New Features
------------------
* ...

Cling as a Library
------------------
* ...

External Dependencies
---------------------
* Upgrade to LLVM rXXX.

Misc
----
* ...

Experimental Features
---------------------
* Code unloading:
* Dynamic Scopes:
* Precompiled Headers: cling now has the ability to generate precompiled headers
cling -x c++-header InputHeader.h -o Output.pch
* Preprocessed Output: cling now has the ability to generate preprocessed output
cling -E -dM will show all preprocessor definitions


Fixed Bugs
----------
<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Standard MarkDown doesn't support neither variables nor <base> --->
[ROOT-XXX](https://sft.its.cern.ch/jira/browse/ROOT-XXX)


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

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.3...master | sort | uniq -c | sort -rn
--->
