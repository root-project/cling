#! /bin/bash

###############################################################################
#
#                           The Cling Interpreter
#
# tools/packaging/get_platform.sh: Script to detect host platform and OS
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

OS=$(uname -o)

if [ "$OS" = "Cygwin" ]; then
  OS="Windows"
elif [ "{$OS}" = "Darwin" ]; then
  OS="Mac OS"
else
  if [ "${OS}" = "GNU/Linux" ] ; then
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
fi

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
