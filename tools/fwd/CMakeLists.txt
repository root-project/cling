set(LLVM_LINK_COMPONENTS
  Support
  Option
  )

add_clang_executable(fwd
  main.cpp
  )

target_link_libraries(fwd
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangDriver
  )

install(TARGETS fwd
  RUNTIME DESTINATION bin)