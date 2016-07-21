#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

# Keep symbols for JIT resolution
set(LLVM_NO_DEAD_STRIP 1)

add_cling_executable(cling
  cling.cpp
  $<TARGET_OBJECTS:obj.clingInterpreter>
  $<TARGET_OBJECTS:obj.clingMetaProcessor>
  $<TARGET_OBJECTS:obj.clingUtils>
  )

set_target_properties(cling
  PROPERTIES ENABLE_EXPORTS 1)

if(MSVC)
  set_target_properties(cling
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif(MSVC)

target_link_libraries(cling
  LLVMSupport
  clingUserInterface
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
