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
  CIFactory.cpp
  DeclCollector.cpp
  DeclExtractor.cpp
  DynamicLookup.cpp
  DynamicExprInfo.cpp
  ExecutionContext.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  InterpreterCallbacks.cpp
  InvocationOptions.cpp
  LookupHelper.cpp
  MultiplexExternalSemaSource.cpp
  RequiredSymbols.cpp
  ReturnSynthesizer.cpp
  StoredValueRef.cpp
  Transaction.cpp
  TransactionTransformer.cpp
  Value.cpp
  ValuePrinter.cpp
  ValuePrinterInfo.cpp
  ValuePrinterSynthesizer.cpp
  )

#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)
