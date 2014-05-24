#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

add_cling_library(clingCtagsExtension
  CtagsWrapper.cpp
  AutoloadCallback.cpp
  readtags.cpp
  TagManager.cpp
  FSUtils.cpp

  LINK_LIBS
  clangBasic
  )
