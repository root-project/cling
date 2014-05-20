#! /bin/bash

###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/indep.sh: Platform independent script with helper functions
# for CPT.
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

function set_version {

  cd ${CLING_SRC_DIR}
  VERSION=$(cat ${CLING_SRC_DIR}/VERSION)

  # If development release, then add revision to the version
  REVISION=$(git log -n 1 --pretty=format:"%H")
  echo "${VERSION}" | grep -qE "dev"
  if [ "${?}" = 0 ]; then
    VERSION="${VERSION}"-"$(echo ${REVISION} | cut -c1-7)"
  fi
}

function compile {
  prefix=${1}
  python=$(type -p python2)
  echo "Create temporary build directory:"
  mkdir -p ${workdir}/builddir
  cd ${workdir}/builddir

  echo "Configuring Cling for compilation"
  ${srcdir}/configure --disable-compiler-version-checks --with-python=${python} --enable-targets=host --prefix=${prefix} --enable-optimized=yes --enable-cxx11

  echo "Building Cling..."
  # TODO: "nproc" program is a part of GNU Coreutils and may not be available on all systems. Use a better solution if needed.
  cores=$(nproc)
  echo "Using ${cores} cores."
  make -j${cores}
  rm -rf ${prefix}
  make install -j${cores}
}

function compile_cygwin {
  # Add code to compile using CMake and MSVC 2012
  :
}

function tarball {
  echo "Compressing ${prefix} to produce a bzip2 tarball..."
  cd ${workdir}
  tar -cjvf $(basename ${prefix}).tar.bz2 -C . $(basename ${prefix})
}
