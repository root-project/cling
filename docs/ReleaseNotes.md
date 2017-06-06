Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 0.4. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-project/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 0.4?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

Major New Features
------------------
* Clang and LLVM optimizations of interpreted code, defaults to -O2 (tweak with
`.O <N>`)
* Support unicode
* Enable C++ modules builds for cling when built with clang
* Functions can be declared at cling's prompt without `.rawInput`

Cling as a Library
------------------
* Add a [simple demo](../tools/demo/cling-demo.cpp) introducing cling as
an interpreter library

Misc
----
* Enable colors at the prompt
* Increase support in printing out objects such as printing of structs and
collections
* Improve performance in code transformations for interactive use
* Improve matching of runtime and build time environments
* Improve stability of the `cpt.py` build tool

Experimental Features
---------------------
* Progress with support on Windows
* Progress with PowerPC 64
* Preprocessed Output: cling now has the ability to generate preprocessed output
cling -E -dM will show all preprocessor definitions at startup

Jupyter
-------
* Support of current Jupyter versions fixing ZMQ communication issues
* Support for c++11, c++14 and c++1z kernels

Fixed Bugs
----------
* Coverity static analysis fixes

[ROOT-7016](https://sft.its.cern.ch/jira/browse/ROOT-7016)
[ROOT-8739](https://sft.its.cern.ch/jira/browse/ROOT-8739)
[ROOT-8696](https://sft.its.cern.ch/jira/browse/ROOT-8696)
[ROOT-8523](https://sft.its.cern.ch/jira/browse/ROOT-8523)
[ROOT-8399](https://sft.its.cern.ch/jira/browse/ROOT-8399)
[ROOT-7354](https://sft.its.cern.ch/jira/browse/ROOT-7354)
[ROOT-8529](https://sft.its.cern.ch/jira/browse/ROOT-8529)
[ROOT-8467](https://sft.its.cern.ch/jira/browse/ROOT-8467)
[ROOT-6539](https://sft.its.cern.ch/jira/browse/ROOT-6539)
[ROOT-8443](https://sft.its.cern.ch/jira/browse/ROOT-8443)
[ROOT-7037](https://sft.its.cern.ch/jira/browse/ROOT-7037)
[ROOT-8443](https://sft.its.cern.ch/jira/browse/ROOT-8443)
[ROOT-8379](https://sft.its.cern.ch/jira/browse/ROOT-8379)
[ROOT-8375](https://sft.its.cern.ch/jira/browse/ROOT-8375)
[ROOT-8392](https://sft.its.cern.ch/jira/browse/ROOT-8392)
[ROOT-7610](https://sft.its.cern.ch/jira/browse/ROOT-7610)
[ROOT-5248](https://sft.its.cern.ch/jira/browse/ROOT-5248)
[ROOT-7857](https://sft.its.cern.ch/jira/browse/ROOT-7857)
[ROOT-8300](https://sft.its.cern.ch/jira/browse/ROOT-8300)
[ROOT-8325](https://sft.its.cern.ch/jira/browse/ROOT-8325)

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.3..master | grep 'ROOT-' | \
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

Frederich Munch (235)
Axel Naumann (191)
Roman Zulak (66)
Vassil Vassilev (63)
Bertrand Bellenot (45)
Philippe Canal (29)
erlanger (7)
Danilo Piparo (4)
Raphael Isemann (3)
Pere Mato (3)
David Abdurachmanov (2)
Sylvain Corlay (1)
Spencer Lyon (1)
Sebastian Uhl (1)
Santiago Castro (1)
PrometheusPi (1)
Pedro Henriques dos Santos Teixeira (1)
Maarten Scholl (1)
Diego Torres Quintanilla (1)
CristinaCristescu (1)
Brian Bockelman (1)
Boris Perovic (1)
Ajith Pandel (1)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.3...master | sort | uniq -c | sort -rn
--->
