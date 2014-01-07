#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set(LLVM_USED_LIBS 
	clangBasic
	)

add_cling_library(clingMetaProcessor
  Display.cpp
  InputValidator.cpp
  MetaLexer.cpp
  MetaParser.cpp
  MetaProcessor.cpp
  MetaSema.cpp)
