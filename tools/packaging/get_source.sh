#! /bin/bash

###############################################################################
#
#                           The Cling Interpreter
#
# tools/packaging/get_source.sh: Script to fetch sources of Cling and vendor
# clones of LLVM and Clang
#
# TODO: Add documentation here, or provide link to documentation
#
# Author: Anirudha Bose <ani07nov@gmail.com>
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#
###############################################################################

# Uncomment the following line to trace the execution of the shell commands
# set -o xtrace

# Fetch the sources for the vendor clone of LLVM
function fetch_llvm {
  # TODO: Change the URL to use the actual Git repo of Cling, rather than Github.
  #       Use "git archive --remote=<url> ..." or similar to remove "curl" as dependency.
  LLVMRevision=$(curl --silent https://raw.githubusercontent.com/ani07nov/cling/master/LastKnownGoodLLVMSVNRevision.txt)
  echo "Last known good LLVM revision is: ${LLVMRevision}"

  if [ -d "${srcdir}" ]; then
    cd "${srcdir}"
    git clean -f -x -d
    git fetch --tags
    git checkout ROOT-patches-r${LLVMRevision}
    git pull origin refs/tags/ROOT-patches-r${LLVMRevision}
  else
    git clone http://root.cern.ch/git/llvm.git "${srcdir}"
    cd "${srcdir}"
    git checkout tags/ROOT-patches-r${LLVMRevision}
  fi
}

# Fetch the sources for the vendor clone of Clang
function fetch_clang {
  if [ -d "${srcdir}/tools/clang" ]; then
    cd "${srcdir}/tools/clang"
    git clean -f -x -d
    git fetch --tags
    git checkout ROOT-patches-r${LLVMRevision}
    git pull origin refs/tags/ROOT-patches-r${LLVMRevision}
  else
    git clone http://root.cern.ch/git/clang.git  "${srcdir}/tools/clang"
    cd "${srcdir}/tools/clang"
    git checkout ROOT-patches-r${LLVMRevision}
  fi
}

# Fetch the sources for Cling
function fetch_cling {
  if [ -d "${srcdir}/tools/cling" ]; then
    cd "${srcdir}/tools/cling"
    git clean -f -x -d
    git fetch --tags
    git checkout ${1}
    git pull origin ${1}
  else
    git clone http://root.cern.ch/git/cling.git  "${srcdir}/tools/cling"
    cd "${srcdir}/tools/cling"
    git checkout ${1}
  fi
}
