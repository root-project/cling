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

function make_nsi {
  cd ${prefix}
  # create installer script
  echo "; Cling setup script ${prefix}" > ${workdir}/cling.nsi
  # installer settings
  cat >> ${workdir}/cling.nsi << EOF
!define APP_NAME "Cling"
!define COMP_NAME "CERN"
!define WEB_SITE "http://cling.web.cern.ch/"
!define VERSION "${VERSION}"
!define COPYRIGHT "Copyright © 2007-2014 by the Authors; Developed by The ROOT Team, CERN and Fermilab"
!define DESCRIPTION "Interactive C++ interpreter"
!define INSTALLER_FILES "${CLING_SRC_DIR}/tools/packaging/windows"
!define INSTALLER_NAME "$(basename ${prefix})-setup.exe"
!define MAIN_APP_EXE "cling.exe"
!define INSTALL_TYPE "SetShellVarContext current"
!define PRODUCT_ROOT_KEY "HKLM"
!define PRODUCT_KEY "Software\Cling"

###############################################################################

VIProductVersion  "\${VERSION}"
VIAddVersionKey "ProductName"  "\${APP_NAME}"
VIAddVersionKey "CompanyName"  "\${COMP_NAME}"
VIAddVersionKey "LegalCopyright"  "\${COPYRIGHT}"
VIAddVersionKey "FileDescription"  "\${DESCRIPTION}"
VIAddVersionKey "FileVersion"  "\${VERSION}"

###############################################################################

SetCompressor /SOLID Lzma
Name "\${APP_NAME}"
Caption "\${APP_NAME}"
OutFile "\${INSTALLER_NAME}"
BrandingText "\${APP_NAME}"
XPStyle on
InstallDir "C:\\Cling\\cling-\${VERSION}"

###############################################################################
; MUI settings
!include "MUI.nsh"

!define MUI_ABORTWARNING
!define MUI_UNABORTWARNING
!define MUI_HEADERIMAGE

; Theme
; Artwork TBA

!insertmacro MUI_PAGE_WELCOME

!define MUI_LICENSEPAGE_TEXT_BOTTOM "The source code for Cling is freely redistributable under the terms of the GNU Lesser General Public License (LGPL) as published by the Free Software Foundation."
!define MUI_LICENSEPAGE_BUTTON "Next >"
!insertmacro MUI_PAGE_LICENSE "${CLING_SRC_DIR}/LICENSE.TXT"

!insertmacro MUI_PAGE_DIRECTORY

!insertmacro MUI_PAGE_INSTFILES

!define MUI_FINISHPAGE_RUN "\$INSTDIR\bin\\\${MAIN_APP_EXE}"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM

!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

###############################################################################

; file section
Section "MainFiles"
EOF

  # Insert the files to be installed
  IFS=$'\n'
  for f in $(find ${prefix} -type d -printf "%P\n"); do
    winf=$(echo $f | sed 's,/,\\\\,g')
    echo " CreateDirectory \"\$INSTDIR\\$winf\"" >> ${workdir}/cling.nsi
    echo " SetOutPath \"\$INSTDIR\\$winf\"" >> ${workdir}/cling.nsi
    find "${prefix}/$f" -maxdepth 1 -type f -printf " File \"%p\"\n" >> ${workdir}/cling.nsi
  done

  cat >> ${workdir}/cling.nsi << EOF

SectionEnd
EOF
}
