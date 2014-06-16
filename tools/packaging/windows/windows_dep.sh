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

function get_nsis {
  box_draw "Check SourceForge project page of NSIS"
  NSIS_VERSION=$(wget -q -O- 'http://sourceforge.net/p/nsis/code/HEAD/tree/NSIS/tags/' | \
    grep '<a href="' | \
    sed -n 's,.*<a href="v\([0-9]\)\([^"]*\)".*,\1.\2,p' | \
    tail -1)
  echo "Latest version of NSIS is: "${NSIS_VERSION}
  box_draw "Download NSIS compiler (makensis.exe)"
  wget "http://sourceforge.net/projects/nsis/files/NSIS%203%20Pre-release/${NSIS_VERSION}/nsis-${NSIS_VERSION}.zip"
  unzip -d ${workdir}/install_tmp nsis-${NSIS_VERSION}.zip
  chmod -R 775 ${workdir}/install_tmp/nsis-${NSIS_VERSION}
}

function make_nsi {
  box_draw "Generating cling.nsi"
  cd ${CLING_SRC_DIR}
  VIProductVersion=$(git describe --match v* --abbrev=0 --tags | head -n 1)
  cd ${workdir}
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
!define INSTALLER_NAME "$(basename ${prefix})-setup.exe"
!define MAIN_APP_EXE "cling.exe"
!define INSTALL_TYPE "SetShellVarContext current"
!define PRODUCT_ROOT_KEY "HKLM"
!define PRODUCT_KEY "Software\Cling"

###############################################################################

VIProductVersion  "${VIProductVersion/v/}.0.0"
VIAddVersionKey "ProductName"  "\${APP_NAME}"
VIAddVersionKey "CompanyName"  "\${COMP_NAME}"
VIAddVersionKey "LegalCopyright"  "\${COPYRIGHT}"
VIAddVersionKey "FileDescription"  "\${DESCRIPTION}"
VIAddVersionKey "ProductVersion"  "\${VERSION}"

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
!define MUI_ICON "$(cygpath --windows --absolute ${CLING_SRC_DIR}/tools/packaging/windows/ROOT.ico)"
!define MUI_UNICON "$(cygpath --windows --absolute ${workdir}/install_tmp/nsis-${NSIS_VERSION}/Contrib/Graphics/Icons/orange-uninstall.ico)"

!insertmacro MUI_PAGE_WELCOME

!define MUI_LICENSEPAGE_TEXT_BOTTOM "The source code for Cling is freely redistributable under the terms of the GNU Lesser General Public License (LGPL) as published by the Free Software Foundation."
!define MUI_LICENSEPAGE_BUTTON "Next >"
!insertmacro MUI_PAGE_LICENSE "$(cygpath --windows --absolute ${CLING_SRC_DIR}/LICENSE.TXT)"

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
    for p in $(find "${prefix}/$f" -maxdepth 1 -type f); do
      echo " File \"$(cygpath --windows --absolute $p)\"" >> ${workdir}/cling.nsi
    done
  done

  cat >> ${workdir}/cling.nsi << EOF

SectionEnd

Section make_uninstaller
 ; Write the uninstall keys for Windows
 SetOutPath "\$INSTDIR"
 WriteRegStr HKLM "Software\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "DisplayName" "Cling"
 WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "UninstallString" "\$INSTDIR\uninstall.exe"
 WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "NoModify" 1
 WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Cling" "NoRepair" 1
 WriteUninstaller "uninstall.exe"
SectionEnd

; start menu
# TODO: This is currently hardcoded.
Section "Shortcuts"

 CreateDirectory "\$SMPROGRAMS\\Cling"
 CreateShortCut "\$SMPROGRAMS\\Cling\\Uninstall.lnk" "\$INSTDIR\\uninstall.exe" "" "\$INSTDIR\\uninstall.exe" 0
 CreateShortCut "\$SMPROGRAMS\Cling\\Cling.lnk" "\$INSTDIR\\bin\\cling.exe" "" "\$INSTDIR\\$ICON" 0
 CreateDirectory "\$SMPROGRAMS\\Cling\\Documentation"
 CreateShortCut "\$SMPROGRAMS\\Cling\\Documentation\\Cling (PS).lnk" "\$INSTDIR\\docs\\llvm\\ps\\cling.ps" "" "" 0
 CreateShortCut "\$SMPROGRAMS\\Cling\\Documentation\\Cling (HTML).lnk" "\$INSTDIR\\docs\\llvm\\html\\cling\\cling.html" "" "" 0

SectionEnd

Section "Uninstall"

 DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Cling"
 DeleteRegKey HKLM "Software\Cling"

 ; Remove shortcuts
 Delete "\$SMPROGRAMS\Cling\*.*"
 Delete "\$SMPROGRAMS\Cling\Documentation\*.*"
 Delete "\$SMPROGRAMS\Cling\Documentation"
 RMDir "\$SMPROGRAMS\Cling"

EOF

# insert dir list (backwards order) for uninstall files
  for f in $(find ${prefix} -depth -type d -printf "%P\n"); do
    winf=$(echo $f | sed 's,/,\\\\,g')
    echo " Delete \"\$INSTDIR\\$winf\\*.*\"" >> ${workdir}/cling.nsi
    echo " RmDir \"\$INSTDIR\\$winf\"" >> ${workdir}/cling.nsi
  done

# last bit of the uninstaller
  cat >> ${workdir}/cling.nsi << EOF
 Delete "\$INSTDIR\*.*"
 RmDir "\$INSTDIR"
SectionEnd
EOF
}

function build_nsis {
  box_draw "Building NSIS executable from cling.nsi"
  ${workdir}/install_tmp/nsis-${NSIS_VERSION}/makensis.exe -V3 ${workdir}/cling.nsi
}
