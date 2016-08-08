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
    clangFrontendTool
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
    clangFrontendTool
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
  set_property(TARGET cling APPEND_STRING PROPERTY LINK_FLAGS
    " /EXPORT:??_7type_info@@6B@ /EXPORT:?__type_info_root_node@@3U__type_info_node@@A")
endif(MSVC)

target_link_libraries(cling
  ${LIBS}
  )

install(TARGETS cling
  RUNTIME DESTINATION bin)
