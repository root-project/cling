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
  if [ "$(command -v ${1})" = "" ]; then
    printf "%-10s\t\t[NOT INSTALLED]\n" "${1}"
  else
    printf "%-10s\t\t[OK]\n" "${1}"
  fi
}
