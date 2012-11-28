set(LLVM_USED_LIBS 
	clangBasic
	)

add_cling_library(clingMetaProcessor
  Display.cpp
  MetaProcessor.cpp
  InputValidator.cpp)
