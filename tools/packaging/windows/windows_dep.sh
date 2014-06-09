#! /bin/bash

###############################################################################
#
#                           The Cling Interpreter
#
# Cling Packaging Tool (CPT)
#
# tools/packaging/windows/windows_dep.sh: Script with helper functions for
# Windows (Cygwin) platform.
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

function check_cygwin {
  # Check for Cygwin
  if [ "${1}" = "cygwin" ]; then
    printf "%-10s\t\t[OK]\n" "${1}"

  # Check for Microsoft Visual Studio 11.0
  elif [ "${1}" = "msvc" ]; then
    cmd.exe /C REG QUERY "HKEY_CLASSES_ROOT\VisualStudio.DTE.11.0" | grep -qE "ERROR"
    if [ "${?}" = 0 ]; then
      printf "%-10s\t\t[NOT INSTALLED]\n" "${1}"
    else
      printf "%-10s\t\t[OK]\n" "${1}"
    fi

  # Check for other tools
  elif [ "$(command -v ${1})" = "" ]; then
    printf "%-10s\t\t[NOT INSTALLED]\n" "${1}"
  else
    printf "%-10s\t\t[OK]\n" "${1}"
  fi
}

function build_nsis {
  box_draw "Check SourceForge project page of NSIS"
  NSIS_VERSION=$(wget -q -O- 'http://sourceforge.net/p/nsis/code/HEAD/tree/NSIS/tags/' | \
    grep '<a href="' | \
    sed -n 's,.*<a href="v\([0-9]\)\([^"]*\)".*,\1.\2,p' | \
    tail -1)
  echo "Latest version of NSIS is: "${NSIS_VERSION}
  box_draw "Download NSIS compiler (makensis.exe)"
  wget "http://sourceforge.net/projects/nsis/files/NSIS%203%20Pre-release/${NSIS_VERSION}/nsis-${NSIS_VERSION}.zip"
  unzip -d ${workdir}/install_tmp nsis-${NSIS_VERSION}.zip
}
