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
  ExecutionContext.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  InterpreterCallbacks.cpp
  InvocationOptions.cpp
  IRDumper.cpp
  LookupHelper.cpp
  NullDerefProtectionTransformer.cpp
  RequiredSymbols.cpp
  ReturnSynthesizer.cpp
  RuntimeException.cpp
  StoredValueRef.cpp
  Transaction.cpp
  TransactionTransformer.cpp
  Value.cpp
  ValuePrinter.cpp
  ValuePrinterInfo.cpp
  ValuePrinterSynthesizer.cpp
  )
  
set_source_files_properties(RuntimeException.cpp COMPILE_FLAGS -fexceptions)

#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)
