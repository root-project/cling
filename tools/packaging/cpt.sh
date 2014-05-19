#! /bin/bash

###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/cpt.sh: Main script which calls other helper scripts to
# compile and package Cling for multiple platforms.
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

# TODO: Change workdir to a suitable path on your system (or Electric Commander)
workdir=~/ec/build
srcdir=${workdir}/cling-src
CLING_SRC_DIR=${srcdir}/tools/cling

# Import helper scripts here
source $(dirname ${0})/get_source.sh
source $(dirname ${0})/build.sh
source $(dirname ${0})/get_platform.sh
source $(dirname ${0})/debian/debianize.sh

function usage {
  echo ""
  echo "Cling Packaging Tool"
  echo ""
  echo "Usage: ./cpt.sh {arg}"
  echo -e "    -h, --help\t\t\tDisplay this help and exit"
  echo -e "    --check-requirements\tCheck if packages required by the script are installed"
  echo -e "    --current-dev={tar,deb}\tCompile the latest development snapshot and produce a tarball/Debian package"
  echo -e "    --last-stable-tarball\tCompile the last stable snapshot and produce a tarball"
  echo -e "    --last-stable-deb\t\tCompile the last stable snapshot and produce a Debian package"
  echo -e "    --tarball-tag={tag}\t\tCompile the snapshot of a given tag and produce a tarball"
  echo -e "    --deb-tag={tag}\t\tCompile the snapshot of a given tag and produce a Debian package"
}

while [ "${1}" != "" ]; do
  if [ "${#}" != 1 ]; then
    echo "Error: script can handle only one switch at a time"
    usage
    exit
  fi

  PARAM=$(echo ${1} | awk -F= '{print $1}')
  VALUE=$(echo ${1} | awk -F= '{print $2}')

  case ${PARAM} in
    -h | --help)
        usage
        exit
        ;;
    --check-requirements)
        echo "Checking if required softwares are available on this system..."
        if [ ${DIST} = "Ubuntu" ]; then
          check_ubuntu git
          check_ubuntu curl
          check_ubuntu debhelper
          check_ubuntu devscripts
          check_ubuntu gnupg
          check_ubuntu python
          echo -e "\nYou are advised to make sure you have the \"latest\" versions of the above packages installed."
          echo -e "Update the required packages by:"
          echo -e "  sudo apt-get update"
          echo -e "  sudo apt-get install git curl debhelper devscripts gnupg python"
          exit
        elif [ ${OS} = "Cygwin" ]; then
          :
        fi

        ;;
    --current-dev)
        fetch_llvm
        fetch_clang
        fetch_cling master
        set_version
        if [ ${VALUE} = "tar" ]; then
          compile ${workdir}/cling-$(get_DIST)-$(get_REVISION)-$(get_BIT)bit-${VERSION}
          tarball
        elif [ ${VALUE} = "deb" ]; then
          compile ${workdir}/cling-${VERSION}
          tarball_deb
          debianize
          cleanup_deb
        fi
        ;;
    --last-stable-tarball)
        fetch_llvm
        fetch_clang
        cd ${CLING_SRC_DIR}
        fetch_cling $(git describe --match v* --abbrev=0 --tags | head -n 1)
        set_version
        compile ${workdir}/cling-$(get_DIST)-$(get_REVISION)-$(get_BIT)bit-${VERSION}
        tarball
        ;;
    --last-stable-deb)
        fetch_llvm
        fetch_clang
        cd ${CLING_SRC_DIR}
        fetch_cling $(git describe --match v* --abbrev=0 --tags | head -n 1)
        VERSION=$(git describe --match v* --abbrev=0 --tags | head -n 1 | sed s/v//g)
        compile ${workdir}/cling-${VERSION}
        tarball_deb
        debianize
        cleanup_deb
        ;;
    --tarball-tag)
        fetch_llvm
        fetch_clang
        fetch_cling ${VALUE}
        VERSION=$(echo ${VALUE} | sed s/v//g)
        compile ${workdir}/cling-$(get_DIST)-$(get_REVISION)-$(get_BIT)bit-${VERSION}
        tarball
        ;;
    --deb-tag)
        fetch_llvm
        fetch_clang
        fetch_cling ${VALUE}
        VERSION=$(echo ${VALUE} | sed s/v//g)
        compile ${workdir}/cling-${VERSION}
        tarball_deb
        debianize
        cleanup_deb
        ;;
    *)
        echo "Error: unknown parameter \"${PARAM}\""
        usage
        exit 1
        ;;
  esac
  shift
done
