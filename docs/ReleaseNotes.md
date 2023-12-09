Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 1.0. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main
[Cling web page](https://rawgit.com/root-project/cling/master/www/index.html),
this document applies to the *next* release, not the current one.

What's New in Cling 1.0?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM 13 and reduce the accumulated technical debt in our local fork
* Require C++14


Major New Features
------------------
* Improve C++ modules support for C++20 and Windows
* Improve performance by allowing most of cling::Value to inline
* Support profiling/debugging interpreted/JITted via `CLING_DEBUG` and
  `CLING_PROFILE`
* Partially support Apple M1
* Improve transaction unloader for templates
* Always emit weak symbols on Windows
* Support RPATH in dyld



Misc
----
* Improve user experience on terminal:
  - Move between words with Ctrl+{Left,Right}
  - Clear the terminal screen on Ctrl+L
  - Bind Ctrl+Del to kill next word
  - Do not assign ESC a special meaning on history search mode
  - Implement `.help edit` to show line editor keybindings
  - .x does not crash if no argument is given
* Support setting cmd history file by `${CLING_HISTFILE}` and its size with
  `${CLING_HISTSIZE}`
* Improve `.help` and `.class` commands
* Improve CUDA support on Visual Studio
* Improve symbol location diagnostics using the dyld infrastructure
* Better support of ppc


Fixed Bugs
----------
[ROOT-10962](https://sft.its.cern.ch/jira/browse/ROOT-10962)
[ROOT-10484](https://sft.its.cern.ch/jira/browse/ROOT-10484)
[ROOT-9687](https://sft.its.cern.ch/jira/browse/ROOT-9687)
[ROOT-9202](https://sft.its.cern.ch/jira/browse/ROOT-9202)
[ROOT-7775](https://sft.its.cern.ch/jira/browse/ROOT-7775)
[ROOT-7016](https://sft.its.cern.ch/jira/browse/ROOT-7016)
[ROOT-6095](https://sft.its.cern.ch/jira/browse/ROOT-6095)
[ROOT-5971](https://sft.its.cern.ch/jira/browse/ROOT-5971)
[ROOT-5219](https://sft.its.cern.ch/jira/browse/ROOT-5219)

[GH-454](https://github.com/root-project/cling/issues/454)
[GH-444](https://github.com/root-project/cling/issues/444)
[GH-440](https://github.com/root-project/cling/issues/440)
[GH-436](https://github.com/root-project/cling/issues/436)

[GH-13815](https://github.com/root-project/root/issues/13815)
[GH-12779](https://github.com/root-project/root/issues/12779)
[GH-12457](https://github.com/root-project/root/issues/12457)
[GH-12455](https://github.com/root-project/root/issues/12455)
[GH-13429](https://github.com/root-project/root/issues/13429)
[GH-12409](https://github.com/root-project/root/issues/12409)
[GH-12294](https://github.com/root-project/root/issues/12294)
[GH-12151](https://github.com/root-project/root/issues/12151)
[GH-11937](https://github.com/root-project/root/issues/11937)
[GH-11933](https://github.com/root-project/root/issues/11933)
[GH-11329](https://github.com/root-project/root/issues/11329)
[GH-11927](https://github.com/root-project/root/issues/11927)
[GH-10209](https://github.com/root-project/root/issues/10209)
[GH-10182](https://github.com/root-project/root/issues/10182)
[GH-10180](https://github.com/root-project/root/issues/10180)
[GH-10137](https://github.com/root-project/root/issues/10137)
[GH-10136](https://github.com/root-project/root/issues/10136)
[GH-10135](https://github.com/root-project/root/issues/10135)
[GH-10133](https://github.com/root-project/root/issues/10133)
[GH-10057](https://github.com/root-project/root/issues/10057)
[GH-9850](https://github.com/root-project/root/issues/9850)
[GH-9697](https://github.com/root-project/root/issues/9697)
[GH-9664](https://github.com/root-project/root/issues/9664)
[GH-9449](https://github.com/root-project/root/issues/9449)
[GH-8499](https://github.com/root-project/root/issues/8499)
[GH-8389](https://github.com/root-project/root/issues/8389)
[GH-8304](https://github.com/root-project/root/issues/8304)
[GH-8292](https://github.com/root-project/root/issues/8292)
[GH-8157](https://github.com/root-project/root/issues/8157)
[GH-8141](https://github.com/root-project/root/issues/8141)
[GH-7541](https://github.com/root-project/root/issues/7541)
[GH-7483](https://github.com/root-project/root/issues/7483)
[GH-7366](https://github.com/root-project/root/issues/7366)

<!---Get release bugs
git log v0.9..master | grep 'ROOT-' | sed -E \
  's,^.*(ROOT-[0-9]+).*$,[\1]\(https://sft.its.cern.ch/jira/browse/\1\),' | \
  sort | uniq
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

Vassil Vassilev (131)
Jonas Hahnfeld (71)
Axel Naumann (66)
Javier Lopez-Gomez (48)
saisoma123 (29)
ferdymercury (12)
Jiang Yi (11)
Bertrand Bellenot (10)
Sergey Linev (9)
Stephan Lachnit (4)
Guilherme Amadio (4)
Surya Somayyajula (3)
Simeon Ehrig (3)
Stefan Gränitz (2)
Maksymilian Graczyk (2)
Garima Singh (2)
Duncan Ogilvie (2)
Baidyanath Kundu (2)
Sara Bellei (1)
Oksana Shadura (1)
Mikolaj Krzewicki (1)
Mattias Ellert (1)
Karel Balej (1)
Jonas Rembser (1)
Enrico Guiraud (1)
Danilo Piparo (1)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.9...master | sort | uniq -c | sort -rn |\
  sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
