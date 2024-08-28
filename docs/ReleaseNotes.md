Introduction
============

This document contains the release notes for the interactive C++ interpreter
Cling, release 1.1. Cling is built on top of [Clang](http://clang.llvm.org) and
[LLVM](http://llvm.org>) compiler infrastructure. Here we
describe the status of Cling in some detail, including major
improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout or the main Cling
web page, this document applies to the *next* release, not the current one.

What's New in Cling 1.1?
========================

Some of the major new features and improvements to Cling are listed
here. Generic improvements to Cling as a whole or to its underlying
infrastructure are described first.

External Dependencies
---------------------
* Upgrade to LLVM LLVM16 and further reduce our technical debt
* Require C++17

Major New Features
------------------
* Support ppc64 in jitlink
* Support LLVM plugins
* Improve the modulemap handling on Darwin

Fixed Bugs
----------
[442](https://github.com/root-project/cling/issues/442)
[14593](https://github.com/root-project/root/issues/14593)
[16219](https://github.com/root-project/root/issues/16219)
[11190](https://github.com/root-project/root/issues/11190)
[14964](https://github.com/root-project/root/issues/14964)
[16121](https://github.com/root-project/root/issues/16121)

Special Kudos
=============
This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

Jonas Hahnfeld (53)
Devajith Valaparambil Sreeramaswamy (18)
Vassil Vassilev (10)
Bertrand Bellenot (6)
ferdymercury (4)
Devajth Valaparambil Sreeramaswamy (3)
Yong Gyu Lee (2)
Jonas Rembser (2)
Vincenzo Eduardo Padulano (1)
saisoma123 (1)
Olivier Couet (1)
Maxim Cournoyer (1)
LiAuTraver (1)
Kai Luo (1)
Devajith (1)
Danilo Piparo (1)
Axel Naumann (1)
