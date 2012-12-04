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
