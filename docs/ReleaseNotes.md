Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.6. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-project/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 0.6?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM r302975.

Misc
----
* Improve the diagnostics for lambdas copy captures on global scope.
* Various optimizations in cling runtime such as outlining of `Evaluate`
  functions.

Experimental Features
---------------------
* Advance the C++ modules support in cling

Fixed Bugs
----------

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.5..master | grep 'ROOT-' | \
  s,^.*(ROOT-[0-9]+).*$,[\1]\(https://sft.its.cern.ch/jira/browse/\1\),' | uniq
--->
<!---Standard MarkDown doesn't support neither variables nor <base>
[ROOT-XXX](https://sft.its.cern.ch/jira/browse/ROOT-XXX)
--->

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

Axel Naumann (68)
Frederich Munch (62)
Vassil Vassilev (28)
Raphael Isemann (21)
Bertrand Bellenot (10)
Roman Zulak (9)
Philippe Canal (4)
Danilo Piparo (3)
gouarin (1)
Yuki Yamaura (1)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.5...master | sort | uniq -c | sort -rn
--->
