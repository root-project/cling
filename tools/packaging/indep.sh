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

function platform_init {
  OS=$(uname -o)

  if [ "${OS}" = "Cygwin" ]; then
    DIST="Win"

  elif [ "{$OS}" = "Darwin" ]; then
    OS="Mac OS"

  elif [ "${OS}" = "GNU/Linux" ] ; then
    if [ -f /etc/redhat-release ] ; then
      DistroBasedOn='RedHat'
      DIST=$(cat /etc/redhat-release |sed s/\ release.*//)
      PSEUDONAME=$(cat /etc/redhat-release | sed s/.*\(// | sed s/\)//)
      REV=$(cat /etc/redhat-release | sed s/.*release\ // | sed s/\ .*//)
    elif [ -f /etc/debian_version ] ; then
      DistroBasedOn='Debian'
      DIST=$(cat /etc/lsb-release | grep '^DISTRIB_ID' | awk -F=  '{ print $2 }')
      PSEUDONAME=$(cat /etc/lsb-release | grep '^DISTRIB_CODENAME' | awk -F=  '{ print $2 }')
      REV=$(cat /etc/lsb-release | grep '^DISTRIB_RELEASE' | awk -F=  '{ print $2 }')
    fi
  fi

  if [ "${DIST}" = "" ]; then
    DIST="N/A"
  fi

  if [ "${DistroBasedOn}" = "" ]; then
    DistroBasedOn="N/A"
  fi

  if [ "${PSEUDONAME}" = "" ]; then
    PSEUDONAME="N/A"
  fi

  if [ "${REV}" = "" ]; then
    REV="N/A"
  fi
}

function get_OS {
  printf "%s" "${OS}"
}

function get_DIST {
  printf "%s" "${DIST}"
}

function get_DistroBasedOn {
  printf "%s" "${DistroBasedOn}"
}
function get_PSEUDONAME {
  printf "%s" "${PSEUDONAME}"
}
function get_REVISION {
  printf "%s" "${REV}"
}

function get_BIT {
  printf "%s" "$(getconf LONG_BIT)"
}

# Helper functions to prettify text like that in Debian Build logs
function box_draw_header {
  msg="cling ($(uname -m))$(date)"
  spaces_no=$(echo "80 $(echo ${msg} | wc -m)" | awk '{printf "%d", $1 - $2 - 4}')
  spacer=$(head -c ${spaces_no} < /dev/zero | tr '\0' ' ')
  if [ ${OS} = "Cygwin" ]; then
    msg="cling ($(uname -m))${spacer}$(date)"
  else
    msg="cling ($(uname -m))${spacer} $(date)"
  fi
  echo "\
╔══════════════════════════════════════════════════════════════════════════════╗
║ ${msg} ║
╚══════════════════════════════════════════════════════════════════════════════╝"
}

function box_draw {
  msg=${1}
  spaces_no=$(echo "80 $(echo ${msg} | wc -m)" | awk '{printf "%d", $1 - $2 - 3}')
  spacer=$(head -c ${spaces_no} < /dev/zero | tr '\0' ' ')
  echo "\
┌──────────────────────────────────────────────────────────────────────────────┐
│ ${msg}${spacer} │
└──────────────────────────────────────────────────────────────────────────────┘"
}

# Fetch the sources for the vendor clone of LLVM
function fetch_llvm {
  box_draw "Fetch source files"
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
  box_draw "Set Cling version"
  cd ${CLING_SRC_DIR}
  VERSION=$(cat ${CLING_SRC_DIR}/VERSION)

  # If development release, then add revision to the version
  REVISION=$(git log -n 1 --pretty=format:"%H")
  echo "${VERSION}" | grep -qE "dev"
  if [ "${?}" = 0 ]; then
    VERSION="${VERSION}"-"$(echo ${REVISION} | cut -c1-7)"
  fi
  echo "Version: ${VERSION}"
  if [ ${REVISION} != "" ]; then
    echo "Revision: ${REVISION}"
  fi
}

function compile {
  prefix=${1}
  python=$(type -p python)
  # TODO: "nproc" program is a part of GNU Coreutils and may not be available on all systems. Use a better solution if needed.
  cores=$(nproc)

  # Cleanup previous installation directory if any
  rm -Rf ${prefix}
  mkdir -p ${workdir}/builddir
  cd ${workdir}/builddir

  if [ "${OS}" = "Cygwin" ]; then
    box_draw "Configuring Cling with CMake and generating Visual Studio 11 project files"
    cmake -G "Visual Studio 11" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(cygpath --windows --absolute ${workdir}/install_tmp) ../$(basename ${srcdir})

    box_draw "Building Cling (using ${cores} cores)"
    cmake --build . --target clang --config Release
    cmake --build . --target cling --config Release
    box_draw "Install compiled binaries to prefix (using ${cores} cores)"
    cmake --build . --target INSTALL --config Release
  else
    box_draw "Configuring Cling for compilation"
    ${srcdir}/configure --disable-compiler-version-checks --with-python=${python} --enable-targets=host --prefix=${workdir}/install_tmp --enable-optimized=yes --enable-cxx11

    box_draw "Building Cling (using ${cores} cores)"
    make -j${cores}
  fi
}

function install_prefix {
    if [ "${OS}" = "Cygwin" ]; then
      box_draw "Install compiled binaries to prefix (using ${cores} cores)"
      cmake --build . --target INSTALL --config Release
    else
      box_draw "Install compiled binaries to prefix (using ${cores} cores)"
      make install -j${cores}
    fi

    for f in $(find ${workdir}/install_tmp -type f -printf "%P\n"); do
      grep -q $(basename $f)[[:space:]] $(dirname ${0})/dist-files.mk
      if [ ${?} = 0 ]; then
        mkdir -p ${prefix}/$(dirname $f)
        cp ${workdir}/install_tmp/$f ${prefix}/$f
      fi
    done
}

function test_cling {
  box_draw "Run Cling test suite"
  if [ ${OS} != "Cygwin" ]; then
    cd ${workdir}/builddir/tools/cling
    make test
  fi
}

function tarball {
  box_draw "Compressing binaries to produce a bzip2 tarball"
  cd ${workdir}
  tar -cjvf $(basename ${prefix}).tar.bz2 -C . $(basename ${prefix})
}

function cleanup {
  box_draw "Clean up"
  echo "Remove directory: ${workdir}/builddir"
  rm -Rf ${workdir}/builddir
  echo "Remove directory: ${prefix}"
  rm -Rf ${prefix}
  echo "Remove directory: ${workdir}/install_prefix"
  rm -Rf ${workdir}/install_tmp
}

# Initialize variables with details of the platform and Operating System
platform_init
