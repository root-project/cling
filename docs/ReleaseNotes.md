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

Major New Features
------------------
* Integrate the automatic differentiation library clad as a cling plugin.
* Implement basic plugin support -- cling can load shared libraries which can
  specialize its behavior. It relies on the clang plugin infrastructure.
* Emulate thread local storage (TLS) on the platforms where the JIT does not
  support natively.
* Clang and LLVM optimizations of interpreted code, defaults again to `-O0`

Misc
----
* Optimize cling pointer validity checks.
* Speed up the LookupHelper facilities by introducing a parsing cache.
* Implement Control+C and Control+D support.
* Support printing lambda-dependent types.
* Various minor improvements for C++ modules support:
  * Adjust module cache path;
  * Build the cling runtime into a separate module;
  * Support virtual filesystem overlay files;
* Use COFF object file format on Windows -- fixes symbol lookups.

Experimental Features
---------------------
* Start working on CUDA support

Jupyter
-------
* Provide better diagnostics if cling was not found;
* Find back the kernel if brew install was used;


Fixed Bugs
----------
[ROOT-6967](https://sft.its.cern.ch/jira/browse/ROOT-6967)
[ROOT-7749](https://sft.its.cern.ch/jira/browse/ROOT-7749)
[ROOT-8863](https://sft.its.cern.ch/jira/browse/ROOT-8863)
[ROOT-8897](https://sft.its.cern.ch/jira/browse/ROOT-8897)
[ROOT-8991](https://sft.its.cern.ch/jira/browse/ROOT-8991)
[ROOT-9114](https://sft.its.cern.ch/jira/browse/ROOT-9114)
[ROOT-9377](https://sft.its.cern.ch/jira/browse/ROOT-9377)
[ROOT-9672](https://sft.its.cern.ch/jira/browse/ROOT-9672)
[ROOT-9738](https://sft.its.cern.ch/jira/browse/ROOT-9738)
[ROOT-9789](https://sft.its.cern.ch/jira/browse/ROOT-9789)
[ROOT-9924](https://sft.its.cern.ch/jira/browse/ROOT-9924)
[ROOT-10097](https://sft.its.cern.ch/jira/browse/ROOT-10097)
[ROOT-10221](https://sft.its.cern.ch/jira/browse/ROOT-10221)

<!---Get release bugs
git log v0.5..master | grep 'ROOT-' | sed -E \
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

Vassil Vassilev (94)
Axel Naumann (87)
Simeon Ehrig (25)
Bertrand Bellenot (23)
Yuka Takahashi (17)
Danilo Piparo (13)
Raphael Isemann (5)
Guilherme Amadio (5)
Philippe Canal (4)
Oksana Shadura (4)
Vaibhav Garg (2)
Sylvain Corlay (2)
Saagar Jha (2)
Nikita Ermakov (2)
Dheepak Krishnamurthy (2)
xloem (1)
vagrant (1)
straydragon (1)
simeon (1)
lizhangwen (1)
Wolf Behrenhoff (1)
Nathan Daly (1)
Jason Detwiler (1)
Houkime (1)
Damien L-G (1)
Aleksander Gajewski (1)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.5...master | sort | uniq -c | sort -rn |\
  sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
