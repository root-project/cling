#------------------------------------------------------------------------------
# CLING - the C++ LLVM-based InterpreterG :)
#
# This file is dual-licensed: you can choose to license it under the University
# of Illinois Open Source License or the GNU Lesser General Public License. See
# LICENSE.TXT for details.
#------------------------------------------------------------------------------

# Keep symbols for JIT resolution
set(LLVM_NO_DEAD_STRIP 1)

if(BUILD_SHARED_LIBS)
  set(LIBS
    LLVMSupport
    clingInterpreter
    clingMetaProcessor
    clingUserInterface
    clingUtils
  )
  add_cling_executable(cling
    cling.cpp
  )
else()
  set(LIBS
    LLVMSupport
    clingUserInterface
  )
  add_cling_executable(cling
    cling.cpp
    $<TARGET_OBJECTS:obj.clingInterpreter>
    $<TARGET_OBJECTS:obj.clingMetaProcessor>
    $<TARGET_OBJECTS:obj.clingUtils>
  )
endif(BUILD_SHARED_LIBS)

set_target_properties(cling
  PROPERTIES ENABLE_EXPORTS 1)

if(MSVC)
  set_target_properties(cling
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif(MSVC)

target_link_libraries(cling
  ${LIBS}
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
