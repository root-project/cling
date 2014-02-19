#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  linker
  support
  mc
  executionengine
  target
  jit
  x86info
  x86asmprinter
)

#set(LLVM_USED_LIBS
#  clingWhatever
#)


add_cling_library(clingInterpreter
  ASTDumper.cpp
  ASTNodeEraser.cpp
  AutoSynthesizer.cpp
  CheckEmptyTransactionTransformer.cpp
  CIFactory.cpp
  ClangInternalState.cpp
  DeclCollector.cpp
  DeclExtractor.cpp
  DynamicLibraryManager.cpp
  DynamicLookup.cpp
  DynamicExprInfo.cpp
  IncrementalExecutor.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  InterpreterCallbacks.cpp
  InvocationOptions.cpp
  IRDumper.cpp
  LookupHelper.cpp
  NullDerefProtectionTransformer.cpp
  RequiredSymbols.cpp
  ValueExtractionSynthesizer.cpp
  RuntimeException.cpp
  StoredValueRef.cpp
  Transaction.cpp
  TransactionTransformer.cpp
  Value.cpp
  ValuePrinter.cpp
  ValuePrinterInfo.cpp
  ValuePrinterSynthesizer.cpp
  )
  
if(NOT WIN32)
set_source_files_properties(RuntimeException.cpp COMPILE_FLAGS -fexceptions)
else()
#set_source_files_properties(RuntimeException.cpp COMPILE_FLAGS " /EHsc ")
# the line above doesn't work, and it gives the following warnings:
# cl : Command line warning D9025: overriding '/EHs' with '/EHs-'
# cl : Command line warning D9025: overriding '/EHc' with '/EHc-'
# so let's change the global compiler flags instead:
string(REPLACE "/EHs-c-" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)

add_file_dependencies(${CMAKE_CURRENT_SOURCE_DIR}/CIFactory.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/cling-compiledata.h)

# Remove all -I from CMAKE_CXX_FLAGS
string(REPLACE ";" " " __flags "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-I[^ ]+" "" CMAKE_CXX_FLAGS_NO_I "${__flags}")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cling-compiledata.h
  "#define LLVM_CXX \"${CMAKE_CXX_COMPILER} ${CMAKE_CXX_FLAGS_NO_I} ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}\"
")
