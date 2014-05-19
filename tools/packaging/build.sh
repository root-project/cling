#! /bin/bash

###############################################################################
#
#                           The Cling Interpreter
#
# tools/packaging/build.sh: Script to compile Cling and produce tarballs for
# all platforms
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
