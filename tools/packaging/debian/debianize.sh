#! /bin/bash

# Script to package a Cling tarball into a Debian/Ubuntu archive.
#
# Author: Anirudha Bose (ani07nov@gmail.com)
#
# This file is dual licensed. You may license this software under
# one of the following licenses, marked "UI/NCSAOSL" and "LGPL".
# Read LICENSE.TXT for more details.

# Uncomment the following line to trace the execution of shell commands
# set -o xtrace

if [ "${#}" != 1 ]; then
  echo "Error: incorrect number of arguments"
  exit
fi

if [ "${@}" != *.tar.bz2 ]; then
  echo "Error: expected a path to a valid tarball (bzip2) as argument"
  exit
fi

ABSOLUTE_PATH=$(readlink -f "$@")
TOPDIR=$(dirname "${ABSOLUTE_PATH}")
DIST_FILE=$(basename "${ABSOLUTE_PATH}")

# Extract version of Debian package using SED, or using AWK like I have done
# VERSION=$(echo ${DIST_FILE} | sed 's/.*-//' | sed 's/.tar.bz2//g')
VERSION=$(echo "${DIST_FILE}" | awk -F'[-.]' '{print $6}')

echo "Extracting the tarball.."
tar -xjf "${ABSOLUTE_PATH}"

echo "Renaming directories and tarball according to the Debian Policy.."
mv "${TOPDIR}"/"${DIST_FILE/.tar.bz2//}" "${TOPDIR}"/cling-"${VERSION}"
cp "${ABSOLUTE_PATH}" "${TOPDIR}"/cling_"${VERSION}".orig.tar.bz2

# Can refer to relative paths after this
cd "${TOPDIR}"/cling-"${VERSION}"

# Create directory: debian
mkdir -p debian

echo "Create file: debian/source/format"
mkdir -p debian/source
echo "3.0 (quilt)" > debian/source/format

echo "Create file: debian/cling.install"
cat >> debian/cling.install << EOF
bin/* /usr/bin
docs/* /usr/share/doc
include/* /usr/include
lib/* /usr/lib
share/* /usr/share
EOF

echo "Create file: debian/compact"
# Optimize binary compression
echo "7" > debian/compact

echo "Create file: debian/compat"
echo "9" > debian/compat

echo "Create file: debian/control"
cat >> debian/control << EOF
Source: cling
Section: devel
Priority: optional
Maintainer: Cling Developer Team <cling-dev@cern.ch>
Uploaders: Anirudha Bose <ani07nov@gmail.com>
Build-Depends: debhelper (>= 9.0.0)
Standards-Version: 3.9.5
Homepage: http://cling.web.cern.ch/
Vcs-Git: http://root.cern.ch/git/cling.git
Vcs-Browser: http://root.cern.ch/gitweb?p=cling.git;a=summary

Package: cling
Priority: optional
Architecture: any
Depends: ${shlibs:Depends}, ${misc:Depends}
Description: interactive C++ interpreter
 Cling is a new and interactive C++11 standard compliant interpreter built
 on the top of Clang and LLVM compiler infrastructure. Its advantages over
 the standard interpreters are that it has command line prompt and uses
 Just In Time (JIT) compiler for compilation. Many of the developers
 (e.g. Mono in their project called CSharpRepl) of such kind of software
 applications name them interactive compilers.
 .
 One of Cling's main goals is to provide contemporary, high-performance
 alternative of the current C++ interpreter in the ROOT project - CINT. Cling
 serves as a core component of the ROOT system for storing and analyzing the
 data of the Large Hadron Collider (LHC) experiments. The
 backward-compatibility with CINT is major priority during the development.
EOF

echo "Create file: debian/copyright"
cat >> debian/copyright << EOF
Format: http://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: cling
Source: http://root.cern.ch/gitweb?p=cling.git;a=summary

Files: *
Copyright: 2007-2014 by the Authors
License: LGPL-2.0+
Comment: Developed by The ROOT Team; CERN and Fermilab

Files: debian/*
Copyright: 2014 Anirudha Bose <ani07nov@gmail.com>
License: LGPL-2.0+

License: LGPL-2.0+
 This package is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2 of the License, or (at your option) any later version.
 .
 This package is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.
 .
 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
 .
 On Debian systems, the complete text of the GNU Lesser General
 Public License can be found in "/usr/share/common-licenses/LGPL-2".
Comment: Cling can also be licensed under University of Illinois/NCSA
 Open Source License (UI/NCSAOSL).
 .
 More information here: https://github.com/vgvassilev/cling/blob/master/LICENSE.TXT
EOF

echo "Create file: debian/rules"
cat >> debian/rules << EOF
#!/usr/bin/make -f
# -*- makefile -*-

%:
	dh \$@

override_dh_auto_build:

override_dh_auto_install:
EOF

# The author of the top level changeset is the one who has to sign the Debian package.
echo "Create file: debian/changelog"
cat >> debian/changelog << EOF
cling (${VERSION}-1) unstable; urgency=low

  * [Debian] Update package to version: ${VERSION}
 -- Anirudha Bose <ani07nov@gmail.com>  $(date --rfc-2822)

EOF
echo "Old Changelog:" >> debian/changelog

# NOTE: Adapt according to path to the Git source directory
GIT_DIR="${TOPDIR}"/repos/cling
cd "${GIT_DIR}"
git log $(git rev-list HEAD) --format="  * %s%n%n -- %an <%ae>  %cD%n%n" >> "${TOPDIR}"/cling-"${VERSION}"/debian/changelog
cd -

# Create Debian package
debuild

echo "Moving all newly created files to cling-${VERSION}-1"
mkdir "${TOPDIR}"/cling-"${VERSION}"-1
mv "${TOPDIR}"/cling_"${VERSION}"*.deb "${TOPDIR}"/cling-"${VERSION}"-1
mv "${TOPDIR}"/cling_"${VERSION}"*.changes "${TOPDIR}"/cling-"${VERSION}"-1
mv "${TOPDIR}"/cling_"${VERSION}"*.build "${TOPDIR}"/cling-"${VERSION}"-1
mv "${TOPDIR}"/cling_"${VERSION}"*.dsc "${TOPDIR}"/cling-"${VERSION}"-1
mv "${TOPDIR}"/cling_"${VERSION}"*.debian.tar.gz "${TOPDIR}"/cling-"${VERSION}"-1

echo "Cleaning up redundant file.."
rm "${TOPDIR}"/cling_"${VERSION}"*.orig.tar.bz2
rm -R "${TOPDIR}"/cling-"${VERSION}"

echo "Now exiting.."
