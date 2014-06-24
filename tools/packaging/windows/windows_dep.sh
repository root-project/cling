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
  wget "http://sourceforge.net/projects/nsis/files/NSIS%203%20Pre-release/${NSIS_VERSION}/nsis-${NSIS_VERSION}.zip" -P ${TMP_PREFIX}/nsis
  wget "http://stahlworks.com/dev/unzip.exe" -P ${TMP_PREFIX}/nsis
  chmod 775 ${TMP_PREFIX}/nsis/unzip.exe
  ${TMP_PREFIX}/nsis/unzip.exe $(cygpath -wa ${TMP_PREFIX}/nsis/nsis-${NSIS_VERSION}.zip) -d $(cygpath -wa ${TMP_PREFIX}/nsis/)
  chmod -R 775 ${TMP_PREFIX}/nsis/nsis-${NSIS_VERSION}
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
!define MUI_ICON "$(cygpath -wa ${CLING_SRC_DIR}/tools/packaging/windows/LLVM.ico)"
!define MUI_UNICON "$(cygpath -wa ${TMP_PREFIX}/nsis/nsis-${NSIS_VERSION}/Contrib/Graphics/Icons/orange-uninstall.ico)"

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

Function .onInit
  Call DetectWinVer
  Call CheckPrevVersion
FunctionEnd

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

; Function to detect Windows version and abort if Cling is unsupported in the current platform
Function DetectWinVer
  Push \$0
  Push \$1
  ReadRegStr \$0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" CurrentVersion
  IfErrors is_error is_winnt
is_winnt:
  StrCpy \$1 \$0 1
  StrCmp \$1 4 is_error ; Aborting installation for Windows versions older than Windows 2000
  StrCmp \$0 "5.0" is_error ; Removing Windows 2000 as supported Windows version
  StrCmp \$0 "5.1" is_winnt_XP
  StrCmp \$0 "5.2" is_winnt_2003
  StrCmp \$0 "6.0" is_winnt_vista
  StrCmp \$0 "6.1" is_winnt_7
  StrCmp \$0 "6.2" is_winnt_8
  StrCmp \$1 6 is_winnt_8 ; Checking for future versions of Windows 8
  Goto is_error

is_winnt_XP:
is_winnt_2003:
is_winnt_vista:
is_winnt_7:
is_winnt_8:
  Goto done
is_error:
  StrCpy \$1 \$0
  ReadRegStr \$0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" ProductName
  IfErrors 0 +4
  ReadRegStr \$0 HKLM "SOFTWARE\Microsoft\Windows\CurrentVersion" Version
  IfErrors 0 +2
  StrCpy \$0 "Unknown"
  MessageBox MB_ICONSTOP|MB_OK "This version of Cling cannot be installed on this system. Cling is supported only on Windows NT systems. Current system: \$0 (version: \$1)"
  Abort
done:
  Pop \$1
  Pop \$0
FunctionEnd

; Function to check any previously installed version of Cling in the system
Function CheckPrevVersion
  Push \$0
  Push \$1
  Push \$2
  IfFileExists "\$INSTDIR\bin\cling.exe" 0 otherver
  MessageBox MB_OK|MB_ICONSTOP "Another Cling installation (with the same version) has been detected. Please uninstall it first."
  Abort
otherver:
  StrCpy \$0 0
  StrCpy \$2 ""
loop:
  EnumRegKey \$1 \${PRODUCT_ROOT_KEY} "\${PRODUCT_KEY}" \$0
  StrCmp \$1 "" loopend
  IntOp \$0 \$0 + 1
  StrCmp \$2 "" 0 +2
  StrCpy \$2 "\$1"
  StrCpy \$2 "\$2, \$1"
  Goto loop
loopend:
  ReadRegStr \$1 \${PRODUCT_ROOT_KEY} "\${PRODUCT_KEY}" "Version"
  IfErrors finalcheck
  StrCmp \$2 "" 0 +2
  StrCpy \$2 "\$1"
  StrCpy \$2 "\$2, \$1"
finalcheck:
  StrCmp \$2 "" done
  MessageBox MB_YESNO|MB_ICONEXCLAMATION "Another Cling installation (version \$2) has been detected. It is recommended to uninstall it if you intend to use the same installation directory. Do you want to proceed with the installation anyway?" IDYES done IDNO 0
  Abort
done:
  ClearErrors
  Pop \$2
  Pop \$1
  Pop \$0
FunctionEnd
EOF
}

function build_nsis {
  box_draw "Building NSIS executable from cling.nsi"
  ${TMP_PREFIX}/nsis-${NSIS_VERSION}/nsis/makensis.exe -V3 $(cygpath --windows --absolute ${workdir}/cling.nsi)
}
